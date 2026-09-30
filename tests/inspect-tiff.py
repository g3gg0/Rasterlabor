import sys, struct, zlib, hashlib, json
from pathlib import Path

def inspect(path):
    with path.open('rb') as file:
        head=file.read(16)
        assert head[:8] == b'II+\x00\x08\x00\x00\x00', 'Not a little-endian BigTIFF'
        file.seek(struct.unpack_from('<Q',head,8)[0])
        count=struct.unpack('<Q',file.read(8))[0]
        tags={}
        for _ in range(count):
            tag,kind,n,value=struct.unpack('<HHQQ',file.read(20));tags[tag]=(kind,n,value)
        def values(tag):
            kind,n,v=tags[tag]
            if n == 1: return [v]
            assert kind == 16
            file.seek(v);return struct.unpack('<'+'Q'*n,file.read(8*n))
        width,height=values(256)[0],values(257)[0]
        tw,th=values(322)[0],values(323)[0]
        offsets,lengths=values(324),values(325)
        compression=values(259)[0]
        assert compression == 8
        occupied=0; alpha=0; digest=hashlib.sha256()
        for offset,length in zip(offsets,lengths):
            if not offset:
                assert length==0;continue
            assert offset%8 == 0 and offset+length <= path.stat().st_size
            file.seek(offset);pixels=zlib.decompress(file.read(length))
            assert len(pixels)==tw*th*4
            occupied+=1; a=pixels[3::4];alpha+=len(a)-a.count(0);digest.update(pixels)
        assert occupied>0 and alpha>0
        return dict(file=path.name,width=width,height=height,bytes=path.stat().st_size,
          compression='Deflate',tiles=occupied,nontransparentPixels=alpha,pixelHash=digest.hexdigest())

for pattern in sys.argv[1:]:
    paths = sorted(Path('.').glob(pattern)) if '*' in pattern else [Path(pattern)]
    for path in paths: print(json.dumps(inspect(path)))
