# Kalibrierpaket Version 1

ZIP mit `metadata.json` (UTF-8) und rohen Binaerarrays. Optional enthaelt es
`tracking.json` mit einem zum Video gehoerenden XYR-Pfad. Keine Videodaten im
Standardexport. Kennung `format = "microscope-grid-calibration"`,
`model_version = 1`. Alle Binaerzahlen sind **little-endian**, ohne Header oder
Padding. Speicherordnung C/row-major: `[row, col, component]`; x laeuft am
schnellsten nach dem Komponentenindex. Masken sind Bytes, keine Bitfelder.

## Koordinaten

Ganzzahlige Quellkoordinaten sind Pixelzentren. X nach rechts, Y nach unten.
Die Rasterindizierung folgt derselben Orientierung. In Version 1 wird keine
zusaetzliche Spiegelung oder Rotation angewendet; verarbeitet werden
unrotierte kodierte Pixel. `source_width` und `source_height` sind verbindlich.
Ein Detektions-ROI aendert das Quellkoordinatensystem nicht.

`pixels_per_grid_step = s > 0` ist der feste quadratische Rastermassstab.
`grid_step_mm_or_null = a` ist null bei unbekannter realer Rasterweite.
Dann sind auch `mm_per_pixel` und `dpi` null, sonst `a/s` und `25.4*s/a`.
`reference_frame_id` und `reference_pose = {theta: 0, tx: 0, ty: 0}` fixieren
die gemeinsame starre Koordinatenfreiheit, nicht das Offsetfeld.

## Arrays

| Datei | Datentyp | Form | Bedeutung |
| --- | --- | --- | --- |
| `coefficients.bin` | IEEE Float64 | `[ny,nx,2]` | Splinekoeffizienten dx, dy |
| `forward.bin` | IEEE Float32 | `[source_height,source_width,2]` | Absolute entzerrte Koordinaten U(x,y) |
| `inverseX.bin` | IEEE Float32 | `[output_height,output_width]` | Quell-X fuer jeden Zielpixel |
| `inverseY.bin` | IEEE Float32 | `[output_height,output_width]` | Quell-Y fuer jeden Zielpixel |
| `sourceCoverage.bin` | Uint8 | `[source_height,source_width]` | Anzahl beitragender Trainingsframes, saettigend |
| `valid.bin` | Uint8 | `[output_height,output_width]` | 1 = Inversion, bilinearer Rand und Abdeckung gueltig |
| `numericalValid.bin` | Uint8 | `[output_height,output_width]` | 1 = numerisch invertiert, mit bilinearem Rand |

`metadata.arrays[name]` beschreibt jeweils `file`, `dtype`, `length` und `shape`.
`length` zaehlt skalare Elemente, nicht Bytes. Float32-Maps bleiben
vorzeichenbehaftet. Farbdarstellungen werden nicht gespeichert.

Die redundante Offsetmap ergibt sich exakt aus der gespeicherten Vorwaertsmap:

```text
dx[y,x] = forward[y,x,0] - x
dy[y,x] = forward[y,x,1] - y
```

Der inverse Zielpixel `(u,v)` entspricht der Ebenenposition
`q = output_origin_xy + (u,v)`. Damit:

```text
U(inverseX[v,u], inverseY[v,u]) ~= output_origin_xy + (u,v)
rectified[v,u] = bilinear(raw, inverseX[v,u], inverseY[v,u])
```

Immer `valid` pruefen. Der inverse ungueltige Wert ist -1; Koordinaten ohne
gueltige Maske duerfen nicht verwendet werden. `numericalValid` allein belegt
keine ausreichende Kalibrierabdeckung. Die Maps koennen direkt als zwei
CV_32FC1-Arrays an OpenCV `remap` uebergeben werden; ungueltige Zielpixel danach
mit `valid` maskieren.

Die Abdeckung beruht auf einem 24 x 18 Quellraster, unterschiedlichen aktiven
Trainingsframes und der Naehe zu beobachteten Punkten (0,8 beobachtete
Rasterabstaende). Mindestens drei beitragende Trainingsframes werden fuer
eine gueltige Ausgabe verlangt. Diese Werte sind keine Konfidenzwahrscheinlichkeiten.

## Splinebasis

`spline.basis = "uniform-cardinal-cubic"`, `spacing_xy = [h,h]`,
`index_origin = [-1,-1]`, Komponenten `['dx','dy']`. Version 1 verwendet
denselben Abstand in beiden Richtungen, aber unterschiedliche nx und ny:

```text
nx = floor((source_width  - 1) / h) + 4
ny = floor((source_height - 1) / h) + 4
logical_a = stored_col - 1
logical_b = stored_row - 1
```

Die aeusseren Koeffizienten sind mitoptimiert; keine Nullrandbedingung.
Die vier Basisfunktionen und 16 lokalen Gewichte entsprechen Abschnitt 4 der
[mathematischen Spezifikation](pcb_entzerrung_mathematische_spezifikation.md).
Koeffizienten sind keine Feldsamples. Nicht durch Bildinterpolation der
Kontrollwerte auf ein anderes Gitter uebertragen.

## Weitere Metadaten

- `version`: laufende konsistente Feldversion, getrennt von `model_version`.
- `quality`: `provisional` oder `validated`; eingebautes numerisches Urteil,
  keine Aussage ueber Druckfehler oder die absolute Genauigkeit der Optik.
- `validation_metrics`: Trainings- und Validierungsstatistiken, raeumliche
  Fehlerpunkte, Geometriepruefung und Iterationsinformationen.
- `roundtrip`: separate numerische Inversionskontrolle und gueltiger Anteil.
- `parameters`: Detektor-, ROI-, Raster-, Solver- und Frameauswahlparameter
  der gespeicherten konsistenten Version.
- `optical_configuration`: frei eingegebene Beschreibung des festen Aufbaus.
- `video`: Name als Zuordnungskennung, Dateigroesse, lastModified,
  Aufloesung, Codec, Dauer, Framezahl, Bildratenangabe und PTS-Liste in
  Mikrosekunden. Frame-IDs sind nullbasierte Indizes in Praesentationsreihenfolge.
- `observations`: Frames mit `id`, `timestamp` (Mikrosekunden), `points`,
  `enabled`, `accepted`, `role` (`train`/`validation`) und Ablehnungsgrund.
  Punkte enthalten `x`, `y` in Originalpixeln, ganzzahlige `col`/`row` und
  positive `confidence`. Eine Frame-ID kommt hoechstens einmal vor.
- `poses`: Kalibrierposen mit Winkel `theta` in Radiant und `tx`, `ty` in
  entzerrten Pixeln. Sie werden nicht auf spaetere PCB-Aufnahmen uebertragen.
- `created_at`: Exportzeitpunkt im ISO-8601-Format.
- `tracking.json`: optionaler Tracking-Datensatz mit Modus, Einstellungen und
  Roh-/stabilisierten Kameraposen. X/Y sind entzerrte Pixel relativ zum ersten
  Tracking-Frame, Rotation ist in Radiant angegeben. Patch-Bilddaten werden
  nicht eingebettet.

Geometrisch veraltete UI-Einstellungen ersetzen beim Export nicht stillschweigend
die zu den Maps gehoerenden Parameter und Beobachtungen. Import prueft Format,
Arraygroessen, endliche Zahlen, Koordinaten, Masken, Beobachtungen und metrische
Konsistenz; der Worker prueft anschliessend erneut die Feldgeometrie.

## Spaeteres Stitching

Eine PCB-Pose muss auf die urspruenglichen U-Koordinaten bezogen sein. Fuer
eine Pose `m = Rpcb * U(p) + tpcb` wird am Originalpixel
`p = inverseU(transpose(Rpcb) * (m - tpcb))` abgetastet. Beim Zugriff auf die
gespeicherte inverse Map `output_origin_xy` abziehen. Bei Posen, die stattdessen
auf zugeschnittenen Vorschauen geschaetzt werden, den Ursprung entsprechend
einrechnen. Kalibrierposen des bewegten Rasters sind keine PCB-Posen.