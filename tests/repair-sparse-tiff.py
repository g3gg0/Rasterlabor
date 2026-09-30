import struct,zlib,shutil,json
from pathlib import Path
source=Path('web/exports/20260918_004341-browser')
target=source/'gimp';target.mkdir(exist_ok=True)
for original in sorted(source.glob('*-teil-*.tif')):
 destination=target/original.name
 shutil.copyfile(original,destination)
 with destination.open('r+b') as f:
  f.seek(8);ifd=struct.unpack('<Q',f.read(8))[0];f.seek(ifd);n=struct.unpack('<Q',f.read(8))[0];tags={}
  for _ in range(n):
   p=f.tell();tag,kind,count,value=struct.unpack('<HHQQ',f.read(20));tags[tag]=(count,value,p+12)
  count,off,inline=tags[324];offsets_position=off if count>1 else inline
  count,counts,inline=tags[325];counts_position=counts if count>1 else inline
  f.seek(offsets_position);offsets=list(struct.unpack('<'+'Q'*count,f.read(count*8)))
  f.seek(counts_position);sizes=list(struct.unpack('<'+'Q'*count,f.read(count*8)))
  empty=zlib.compress(bytes(tags[322][1]*tags[323][1]*4))
  repaired=[]
  for i,size in enumerate(sizes):
   if size:continue
   f.seek(0,2);p=f.tell();f.write(bytes((-p)%8));p=f.tell();f.write(empty)
   offsets[i]=p;sizes[i]=len(empty);repaired.append(i)
  f.seek(offsets_position);f.write(struct.pack('<'+'Q'*count,*offsets))
  f.seek(counts_position);f.write(struct.pack('<'+'Q'*count,*sizes))
 print(json.dumps(dict(file=str(destination),transparentTiles=repaired,bytes=destination.stat().st_size)))
shutil.copyfile(source/'zusammenfuegen.sh',target/'zusammenfuegen.sh')
