import asyncio
import os
from asyncio import Future, Task, get_running_loop
from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")


class FuturePool:
    """
    Class representing a pool of async workers to simplify working with async functions that need to be restrained.
    Common use is fetching data from sites, where in order to not influence the website performance, limit on active connections is set.
    """

    class IteratorToAsyncIterator:
        def __init__(self, iterator: Iterator[Awaitable[T]]):
            self.iterator = iterator

        def __aiter__(self):
            return self

        async def __anext__(self) -> U:
            future = next(self.iterator)
            return await future

    def __init__(self, number_of_workers: int = (os.cpu_count() or 1)):
        """ """
        assert number_of_workers > 0, "Number of workers must be a positive number"
        self.number_of_workers = number_of_workers
        self.loop = get_running_loop()
        self.workers_locks = [asyncio.Lock() for _ in range(self.number_of_workers)]
        self.tasks = set[Task]()

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, traceback):
        if self.loop.is_running():
            for lock in self.workers_locks:
                await lock.acquire()
            for task in self.tasks:
                task.cancel()
                try:
                    await task
                except asyncio.InvalidStateError:
                    pass
                except asyncio.CancelledError:
                    pass
            for lock in self.workers_locks:
                lock.release()
        return False

    def _get_iterator_(
        self,
        fn: Callable[[T], Awaitable[U]],
        iterable: Iterable[Iterable[T]],
        ordered: bool,
    ) -> Iterator[Awaitable[U]]:
        iterator = enumerate(iterable)
        futures = list[Future[U]]()
        args = deque[tuple[int, T]]()
        not_finished_futures = deque[Future[U]]()

        def add_task():
            arg_tuple = next(iterator, None)
            if arg_tuple is None:
                return False
            args.append(arg_tuple)
            future = self.loop.create_future()
            futures.append(future)
            if not ordered:
                not_finished_futures.append(future)
            return True

        async def worker(w_id: int):
            while len(args) > 0:
                async with self.workers_locks[w_id]:
                    i, arg = args.popleft()
                    try:
                        result = await fn(*arg)
                        future = (
                            futures[i] if ordered else not_finished_futures.popleft()
                        )
                        future.set_result(result)
                    except asyncio.InvalidStateError:
                        return
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        future = (
                            futures[i] if ordered else not_finished_futures.popleft()
                        )
                        future.set_exception(e)
                    add_task()
            self.tasks.remove(asyncio.current_task())

        def create_workers():
            for w_id in range(self.number_of_workers):
                if not add_task():
                    return
                task = self.loop.create_task(worker(w_id))
                self.tasks.add(task)

        class FutureIterator:
            def __init__(self):
                self.current = 0

            def __iter__(self):
                return self

            def __next__(self) -> Awaitable[U]:
                if len(futures) == 0:
                    create_workers()

                if len(futures) == self.current:
                    if not add_task():
                        raise StopIteration
                future = futures[self.current]
                self.current += 1
                return future

        return FutureIterator()

    def map(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> Awaitable[list[U]]:
        """Parallel equivalent of standard map function. Applies provided fn on each item in iterable
        utilizing number_of_workers workers. Function returns Future of all results.

        Example:
        ``` python
        async with FuturePool(3) as fp:
            result = await fp.map(async_fn, range(3))
        ```

        """
        return self.starmap(fn, ((arg,) for arg in iterable))

    def starmap(
        self, fn: Callable[[Any], Awaitable[U]], iterable: Iterable[Iterable[Any]]
    ) -> Awaitable[list[U]]:
        """
        Like `map()` except that the elements of the iterable are expected to be iterables that are unpacked as arguments.
        Hence an iterable of `[(1,2), (3, 4)]` results in `[func(1,2), func(3,4)]`.

        Example:
        ``` python
        async with FuturePool(3) as fp:
            result = await fp.starmap(async_fn_with_2_args, zip(range(3), range(3)))
        ```
        """

        async def lazy_resolve():
            return [await future for future in self._get_iterator_(fn, iterable, True)]

        return lazy_resolve()

    def imap(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> Iterator[Awaitable[U]]:
        """
        Lazy equivalent of `map()`. Returns iterator of futures.

        Examples:
        ``` python
        async with FuturePool(3) as fp:
            iterator = fp.imap(async_fn, range(10000))
            a = await next(iterator)
            b = await next(iterator)
            c = await next(iterator)

        async with FuturePool(3) as fp:
            for future in fp.imap(async_fn, range(10000)):
                result = await future
                do_sth(result)
        ```
        """

        return self._get_iterator_(fn, ((arg,) for arg in iterable), True)

    def imap_async(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> AsyncIterator[U]:
        """
        Lazy async equivalent of `map()`. Returns async iterator of U.

        Example:
        ``` python
        async with FuturePool(3) as fp:
            async for result in fp.imap_async(async_fn, range(10000)):
                do_sth(result)
        ```
        """

        return FuturePool.IteratorToAsyncIterator(self.imap(fn, iterable))

    def imap_unordered(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> Iterator[Awaitable[U]]:
        """
        The same as `imap()` except that the ordering of the results from the returned iterator should be considered arbitrary.
        (Only when there is only one worker process is the order guaranteed to be 'correct'.)

        Examples:
        ``` python
        async with FuturePool(3) as fp:
            iterator = fp.imap_unordered(async_fn_that_takes_variable_time, range(10000))
            a = await next(iterator) // could be async_fn_that_takes_variable_time(0) or async_fn_that_takes_variable_time(1) or ...
            b = await next(iterator) // could be async_fn_that_takes_variable_time(0) or async_fn_that_takes_variable_time(1) or ...
            c = await next(iterator) // could be async_fn_that_takes_variable_time(0) or async_fn_that_takes_variable_time(1) or ...

        async with FuturePool(3) as fp:
            for future in fp.imap_unordered(async_fn_that_takes_variable_time, range(10000)):
                result = await future // could be async_fn_that_takes_variable_time(0) or async_fn_that_takes_variable_time(1) or ...
                do_sth(result)
        ```
        """
        return self._get_iterator_(fn, ((arg,) for arg in iterable), False)

    def imap_unordered_async(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> AsyncIterator[U]:
        """
        The same as `imap_async()` except that the ordering of the results from the returned iterator should be considered arbitrary.
        (Only when there is only one worker process is the order guaranteed to be 'correct'.)

        Example:
        ``` python
        async with FuturePool(3) as fp:
            async for result in fp.imap_unordered_async(async_fn_that_takes_variable_time, range(10000)):
                // result could be async_fn_that_takes_variable_time(0) or async_fn_that_takes_variable_time(1) or ...
                do_sth(result)
        ```
        """
        return FuturePool.IteratorToAsyncIterator(self.imap_unordered(fn, iterable))

    def starimap(
        self, fn: Callable[[Any], Awaitable[U]], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[Awaitable[U]]:
        """
        Like `imap()` except that the elements of the iterable are expected to be iterables that are unpacked as arguments.

        Example:
        ``` python
        async with FuturePool(3) as fp:
            iterator = fp.starimap(async_fn_with_2_args, zip(range(10000), range(10000)))
            a = await next(iterator)
            b = await next(iterator)
            c = await next(iterator)

        async with FuturePool(3) as fp:
            for future in fp.starimap(async_fn_with_2_args, zip(range(10000), range(10000))):
                result = await future
                do_sth(result)
        ```
        """

        return self._get_iterator_(fn, iterable, True)

    def starimap_async(
        self, fn: Callable[[Any], Awaitable[U]], iterable: Iterable[Iterable[Any]]
    ) -> AsyncIterator[U]:
        """
        Lazy async equivalent of `starimap()`. Returns async iterator of U.

        Example:
        ``` python
        async with FuturePool(3) as fp:
            async for result in fp.starimap_async(async_fn_with_2_args, zip(range(10000), range(10000))):
                do_sth(result)
        ```
        """

        return FuturePool.IteratorToAsyncIterator(self.starimap(fn, iterable))

    def starimap_unordered(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> Iterator[Awaitable[U]]:
        """
        The same as `starimap()` except that the ordering of the results from the returned iterator should be considered arbitrary.
        (Only when there is only one worker process is the order guaranteed to be 'correct'.)

        Examples:
        ``` python
        async with FuturePool(3) as fp:
            iterator = fp.starimap_unordered(async_fn_that_takes_variable_time_with_2_args, zip(range(10000), range(10000)))
            a = await next(iterator) // could be async_fn_that_takes_variable_time_with_2_args(0, 0) or async_fn_that_takes_variable_time_with_2_args(1, 1) or ...
            b = await next(iterator) // could be async_fn_that_takes_variable_time_with_2_args(0, 0) or async_fn_that_takes_variable_time_with_2_args(1, 1) or ...
            c = await next(iterator) // could be async_fn_that_takes_variable_time_with_2_args(0, 0) or async_fn_that_takes_variable_time_with_2_args(1, 1) or ...

        async with FuturePool(3) as fp:
            for future in fp.starimap_unordered(async_fn_that_takes_variable_time_with_2_args, zip(range(10000), range(10000))):
                result = await future // could be async_fn_that_takes_variable_time_with_2_args(0, 0) or async_fn_that_takes_variable_time_with_2_args(1, 1) or ...
                do_sth(result)
        ```
        """
        return self._get_iterator_(fn, iterable, False)

    def starimap_unordered_async(
        self, fn: Callable[[T], Awaitable[U]], iterable: Iterable[T]
    ) -> AsyncIterator[U]:
        """
        The same as `starimap_async()` except that the ordering of the results from the returned iterator should be considered arbitrary.
        (Only when there is only one worker process is the order guaranteed to be 'correct'.)

        Example:
        ``` python
        async with FuturePool(3) as fp:
            async for result in fp.starimap_unordered_async(async_fn_that_takes_variable_time_with_2_args, zip(range(10000), range(10000))):
                // result could be async_fn_that_takes_variable_time_with_2_args(0, 0) or async_fn_that_takes_variable_time_with_2_args(1, 1) or ...
                do_sth(result)
        ```
        """
        return FuturePool.IteratorToAsyncIterator(self.starimap_unordered(fn, iterable))
