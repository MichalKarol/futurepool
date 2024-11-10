# FuturePool [![PyPI - Version](https://img.shields.io/pypi/v/futurepool?style=for-the-badge)](https://pypi.org/project/futurepool/)

FuturePool is a package that introduce known concept of multiprocessing Pool to the async/await world, resulting in async workers pool library. It allows for easy translation from multiprocessing to async/await, while keeping the core principle - specified number of workers. FuturePool allows for more flexible usage by providing starimap/starimap_unordered.

FuturePool was created to handle web scrapping, where in order to not overwhelm website with connections and comply with website requirements, specified number of workers was used. FuturePool was extended to handle generic scenarios and published.

## License
MIT

## Example
To see library docs visit [https://MichalKarol.github.io/futurepool/](https://MichalKarol.github.io/futurepool/).

Example translation from multiprocessing to FuturePool

```python
# multiprocessing
from multiprocessing import Pool
from time import sleep

def pool_fn(i):
    sleep(i)
    return i

with Pool(2) as p:
    result = p.map(pool_fn, range(10))
```

```python
# FuturePool
from futurepool import FuturePool
from asyncio import sleep

async def async_pool_fn(i):
    await sleep(i)
    return i

async with FuturePool(2) as fp:
    result = await fp.map(async_pool_fn, range(10))
```

## Author
Michał Karol <michal.p.karol@gmail.com>  
[Mastodon](https://mastodon.pl/@mkarol)  
[Github](https://github.com/MichalKarol)  