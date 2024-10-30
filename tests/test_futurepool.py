import asyncio
import itertools
from datetime import datetime
from typing import Any, Iterable, TypeVar

import pytest

from futurepool import FuturePool, __version__

T = TypeVar("T")

BASE_TIME = 1
THRESHOLD = 1


async def throwing_async_fn(nbr: int):
    if nbr == 1:
        raise Exception("New exception")
    await asyncio.sleep(BASE_TIME)
    return nbr


async def good_async_fn(nbr: int, _: int = 0):
    await asyncio.sleep(BASE_TIME)
    return nbr


async def unordered_good_async_fn(nbr: int, _: int = 0):
    await asyncio.sleep(BASE_TIME + nbr)
    return nbr


def test_version():
    assert __version__ == "1.0.0"


class SavingIterator:
    def __init__(self, items: Iterable[Any]):
        self.items = items
        self.yields = list[tuple[datetime, Any]]()

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        self.yields.append((datetime.now(), item))
        return item


@pytest.mark.asyncio
async def test_map():
    async with FuturePool(3) as fp:
        future = fp.map(good_async_fn, range(3))
        result = await asyncio.wait_for(future, BASE_TIME + THRESHOLD)
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_map_lazy():
    async with FuturePool(2) as fp:
        it = SavingIterator(iter(range(3)))
        result = await asyncio.wait_for(
            fp.map(good_async_fn, it), 2 * BASE_TIME + THRESHOLD
        )
        assert result == [0, 1, 2]
        assert (it.yields[0][0].timestamp() - it.yields[1][0].timestamp()) < THRESHOLD
        assert (it.yields[2][0] - it.yields[1][0]).seconds == BASE_TIME


@pytest.mark.asyncio
async def test_starmap():
    async with FuturePool(3) as fp:
        result = await asyncio.wait_for(
            fp.starmap(good_async_fn, zip(range(3), range(3))),
            BASE_TIME + THRESHOLD,
        )
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_imap():
    async with FuturePool(3) as fp:
        iterator = fp.imap(good_async_fn, range(10000))
        result = await asyncio.wait_for(
            asyncio.gather(*itertools.islice(iterator, 3)), BASE_TIME + THRESHOLD
        )
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_imap_workers():
    async with FuturePool(3) as fp:
        iterator_1 = fp.imap(good_async_fn, range(9, 0, -1))
        iterator_2 = fp.imap(good_async_fn, range(0, 9, 1))

        async def get_results():
            start = datetime.now()
            a = await next(iterator_1)
            b = await next(iterator_2)
            end = datetime.now()
            return (a, b, (end - start).seconds)

        (a, b, time) = await asyncio.wait_for(get_results(), 4 * BASE_TIME + THRESHOLD)
        assert (a, b) == (9, 0)
        assert time <= 4 * BASE_TIME


@pytest.mark.asyncio
async def test_imap_async():
    async with FuturePool(3) as fp:
        result = []
        async for i in fp.imap_async(good_async_fn, range(10000)):
            result.append(i)
            if len(result) == 3:
                break
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_starimap():
    async with FuturePool(3) as fp:
        iterator = fp.starimap(good_async_fn, zip(range(10000), range(10000)))
        result = await asyncio.wait_for(
            asyncio.gather(*itertools.islice(iterator, 3)), BASE_TIME + THRESHOLD
        )
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_starimap_async():
    async with FuturePool(3) as fp:
        result = []
        async for i in fp.starimap_async(
            good_async_fn, zip(range(10000), range(10000))
        ):
            result.append(i)
            if len(result) == 3:
                break
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_imap_unordered():
    async with FuturePool(3) as fp:
        items = [2, 1, 0]
        iterator = fp.imap_unordered(unordered_good_async_fn, items)
        result = await asyncio.wait_for(
            asyncio.gather(*itertools.islice(iterator, 3)),
            BASE_TIME + max(items[:3]) + THRESHOLD,
        )
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_imap_unordered_single():
    async with FuturePool(1) as fp:
        items = [2, 1, 0]
        iterator = fp.imap_unordered(unordered_good_async_fn, items)
        result = await asyncio.wait_for(
            asyncio.gather(*itertools.islice(iterator, 3)),
            3 * BASE_TIME + sum(items[:3]) + THRESHOLD,
        )
        assert result == [2, 1, 0]


@pytest.mark.asyncio
async def test_imap_unordered_async():
    async with FuturePool(3) as fp:
        result = []
        async for i in fp.imap_unordered_async(unordered_good_async_fn, [2, 1, 0]):
            result.append(i)
            if len(result) == 3:
                break
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_starimap_unordered():
    async with FuturePool(3) as fp:
        items = [2, 1, 0]
        iterator = fp.starimap_unordered(
            unordered_good_async_fn, zip(items, range(10000))
        )
        result = await asyncio.wait_for(
            asyncio.gather(*itertools.islice(iterator, 3)),
            BASE_TIME + max(items[:3]) + THRESHOLD,
        )
        assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_starimap_unordered_async():
    async with FuturePool(3) as fp:
        result = []
        async for i in fp.starimap_unordered_async(
            unordered_good_async_fn, zip([2, 1, 0], range(10000))
        ):
            result.append(i)
            if len(result) == 3:
                break
        assert result == [0, 1, 2]
