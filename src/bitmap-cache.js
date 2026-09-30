export function trimBitmapCache(cache, limit) {
  while (cache.size > limit) {
    const oldest = cache.keys().next().value;
    cache.get(oldest).close();
    cache.delete(oldest);
  }
}
