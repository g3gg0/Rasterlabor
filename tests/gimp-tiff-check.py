from gi.repository import Gimp, Gio, GdkPixbuf, GLib
from pathlib import Path
import json
root=Path('D:/Users/g3gg0/Documents/RevEng/Toniebox/TB2/images/web/exports/20260918_004341-browser')
results=[]
for label,source in [('before',root/'20260918_004341-merge-teil-001.tif'),('after-1',root/'gimp/20260918_004341-merge-teil-001.tif'),('after-2',root/'gimp/20260918_004341-merge-teil-002.tif')]:
 image=Gimp.file_load(Gimp.RunMode.NONINTERACTIVE,Gio.File.new_for_path(str(source)))
 data,width,height,bpp=image.get_thumbnail_data(1024,1024)
 pixels=data.get_data()
 pixbuf=GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes.new(pixels),GdkPixbuf.Colorspace.RGB,bpp==4,8,width,height,width*bpp)
 pixbuf.savev(str(root/'gimp'/('gimp-'+label+'.png')),'png',[],[])
 lower=pixels[(height//2)*width*bpp+3::bpp] if bpp==4 else b''
 result=dict(label=label,width=image.get_width(),height=image.get_height(),bpp=bpp,lowerHalfAlphaPixels=sum(v>0 for v in lower))
 results.append(result);print(json.dumps(result));image.delete()
(root/'gimp/gimp-validation.json').write_text(json.dumps(results,indent=2))
assert results[0]['lowerHalfAlphaPixels']==0
assert all(r['lowerHalfAlphaPixels']>10000 for r in results[1:])
