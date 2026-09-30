from gi.repository import Gimp,Gio,GdkPixbuf,GLib
from pathlib import Path
import json
root=Path('D:/Users/g3gg0/Documents/RevEng/Toniebox/TB2/images/web/exports/20260918_004341-browser/gimp')
image=Gimp.file_load(Gimp.RunMode.NONINTERACTIVE,Gio.File.new_for_path(str(root/'20260918_004341-merge-gesamt.tif')))
data,w,h,bpp=image.get_thumbnail_data(1024,1024)
pixels=data.get_data()
GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes.new(pixels),GdkPixbuf.Colorspace.RGB,bpp==4,8,w,h,w*bpp).savev(str(root/'gimp-gesamt.png'),'png',[],[])
bands=[]
for i in range(4):
 start=(h*i//4)*w*bpp;end=(h*(i+1)//4)*w*bpp
 bands.append(sum(a>0 for a in pixels[start+3:end:bpp]))
result=dict(width=image.get_width(),height=image.get_height(),nontransparentPixelsPerQuarter=bands)
assert (result['width'],result['height'])==(16512,15072)
assert all(n>10000 for n in bands)
(root/'gimp-single-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));image.delete()
