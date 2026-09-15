# Rasterlabor

Lokales Browserwerkzeug zur Mikroskopkalibrierung aus einem bewegten ebenen
Linienraster oder Schachbrett. Es berechnet ein gemeinsames kubisches
B-Spline-Entzerrungsfeld und inverse Resampling-Maps. PCB-Stitching ist nicht
Bestandteil dieser Anwendung.

## Start

Voraussetzung: Node.js 20 oder neuer und aktuelles Microsoft Edge mit WebCodecs.
Die folgenden Befehle im Verzeichnis `web` ausfuehren:

```sh
npm ci
npm run build
npm start
```

Die ausgegebene URL, normalerweise <http://localhost:4173>, in Edge oeffnen.
Bei belegtem Port sucht der Server bis Port 4200 weiter. Er bindet nur an
127.0.0.1. Die HTML-Datei nicht direkt per `file:` oeffnen: Module und Worker
benoetigen den lokalen HTTP-Ursprung.

`npm ci` benoetigt einmalig Zugang zum Paketregister. Danach sind Build,
Anwendung und Verarbeitung offline moeglich. Das bereits gebaute Verzeichnis
`dist` enthaelt JavaScript, Worker, Schrift und Symbole ohne CDN-Abhaengigkeiten.
Fuer einen anderen Offline-Rechner genuegen `dist`, `server.mjs` und Node.js;
dort aus dem gemeinsamen Elternverzeichnis `node server.mjs` starten.
Der Server nimmt keine Uploads an. Videodaten und Ergebnisse bleiben lokal.

## Ablauf

1. Video oeffnen. Es wird noch nichts kalibriert. Aufloesung, Codec,
  Praesentationszeitstempel und Frameindex erscheinen nach der Indexierung.
  Play/Pause, Zeitleiste und die Einzelbildtasten sind davon
   unabhaengig. Einzelbildschritte dekodieren ab dem vorausgehenden Keyframe.
2. Mustertyp waehlen und **Erkennung pruefen**. Optional Rasterabstand,
   Linienradius, Kontrast und Erkennungsbereich anpassen. Spalten/Zeilen sind
   Kreuzungspunkte bzw. innere Schachbrettecken, nicht Zellen. Die automatische
   Abstandsschaetzung setzt den Startwert fuer Zielpixel pro Rasterweite.
   Bei der Kalibrierung bleibt dieser Massstab fest.
  **Image-Patches** verwendet stattdessen die PCB selbst: Nur Bildbereiche mit
  Gradienten in zwei Dimensionen werden gewaehlt; einzelne gerade Kanten werden
  verworfen. Patchgroesse 16 bis 256 px bestimmt den Strukturkontext, der
  Suchradius die maximal erwartete Bewegung zum jeweils naechsten Frame.
  Patch-Tracking benoetigt deshalb fortlaufende Frames; ein Sprung initialisiert
  neue Referenzpatches. Ein metrischer Rastermassstab gilt in diesem Modus nicht.
3. Overlay an mehreren Stellen kontrollieren: eine Linie pro Rasterlinie,
   zusammenhaengende Indizes und keine ausgelassenen Linien. Die gruene
   Phase-1-Anzeige bestaetigt nur eine brauchbare Detektion, keine Kalibrierung.
4. **Kalibrierung starten** verarbeitet jeden Frame des gewaehlten Zeitbereichs
  in Praesentationsreihenfolge, ohne zeitbasierte Ausduennung. Standard:
  Mindestbewegung 0 px (Bewegungsfilter aus), letzte 20 Prozent als
  zusammenhaengender Validierungsabschnitt. Eine positive Mindestbewegung
  schliesst aehnliche Frames nur vom Fit aus, nicht von der Erkennung.
  **Update alle Frames** steuert die Haeufigkeit der Feldneuberechnung,
  nicht die Frameauswahl. Wiederholte fast identische Aufnahmen koennen
  dadurch im Fit staerker gewichtet sein als kurze Bewegungsabschnitte.
5. Pause laesst den laufenden Auftrag bzw. konsistenten Fit zu Ende gehen.
   Fortsetzen setzt die Messframeauswahl fort. **Diesen Frame auswerten** und
   **Naechsten Frame auswerten** sind explizite Einzelaktionen. Reines Navigieren
   fuegt keine Beobachtung hinzu. Eine Frame-ID hat hoechstens einen Messbeitrag.
  Nach Import durchlaeuft Fortsetzen den Bereich erneut, behaelt vorhandene
  Messungen und ergaenzt zuvor uebersprungene Frames. Bei Mindestbewegung 0
  werden zuvor nur wegen geringer Bewegung abgelehnte Frames erneut geprueft.
6. Beobachtung ueber ihre Tabellenzeile erneut ansehen, deaktivieren oder ihre
   Rolle aendern. **Neu fitten** verwendet die aktiven Beobachtungen.
   Geometrieaenderungen markieren den bisherigen Feldstand als veraltet.
7. Feld, Abdeckung, Restfehler und entzerrtes Bild inspizieren. Zoomtasten,
   Mausrad und Ziehen navigieren die Ansichten gemeinsam. Im pausierten,
   entzerrten Bild ergeben zwei Klicks eine Abstandsmessung. Ohne reale
   Rasterweite bleiben die Werte in Pixeln; es werden keine DPI erfunden.
8. Kalibrierung als ZIP speichern. Import funktioniert ohne Video. Zum
  Weiterverarbeiten ein Video mit demselben Dateinamen erneut waehlen.
  Abweichende Dateinamen werden angezeigt, setzen Video, Kalibrierung und
  Beobachtungen aber nicht zurueck. Der Browser speichert keine dauerhafte
  Dateiberechtigung.
9. Mit **Korrigiertes Video speichern** werden alle Frames in PTS-Reihenfolge
  entzerrt und lokal als H.264/MP4 ohne Audiospur gespeichert. Die Verarbeitung
  kann langsamer als die Videolaufzeit sein; Frames werden nicht ausgelassen.

Die R/B-Ansicht zeigt das tatsaechliche Offsetfeld: Null = (128, 0, 128),
positives dx erhoeht Rot, positives dy Blau. Der Darstellungsbereich ist fest
editierbar; Clipping wird ausgewiesen. Schraffur bezeichnet unzureichende
Messabdeckung, keine statistische Unsicherheit. Das optionale Ausblenden einer
gemeinsamen starren Transformation aendert nur die Anzeige, nie den Export.

## Rechenweg

- Decoder-Worker: lokaler MP4-Demux mit MP4Box, WebCodecs, echte PTS-Reihenfolge,
  Dateiname als Videozuordnung, maximal vier gecachte Bilder (Zielbudget 48 MiB).
- Rechen-Worker: Rasterdetektion in Originalpixelkoordinaten, gemeinsame
  SE(2)-Posen und kardinales kubisches B-Spline-Feld, zweidimensionale
  Huber-IRLS-Gewichte, normierte Biegeenergie, duenn besetztes LSQR.
- Referenzpose = Identitaet; ihr Feld ist nicht auf null fixiert. Keine
  frameweisen Homographien, Skalierungen oder Scherungen.
- Geometrische Schrittpruefung vor Uebernahme; dichte Vorwaertsmap und
  triangulierte Newton-Initialisierung fuer die inverse Map. Bilineare Vorschau
  verwendet nur gueltige Zielpixel einschliesslich Interpolationsrand.
- Zurueckgehaltene Frames veraendern das Feld nicht; nur ihre starren Posen
  werden angepasst. Fehlerstatistiken und numerischer Roundtrip sind getrennt.

Details: [mathematische Spezifikation](pcb_entzerrung_mathematische_spezifikation.md)
und [Paketformat](CALIBRATION_FORMAT.md).

## Tests

```sh
npm test
npm run build
```

Die Node-Tests pruefen Basis und Ableitungen, affine Randwerte, Identitaetsfall,
asymmetrische Feldrekonstruktion mit unabhaengigen Testpositionen, starre
Referenzpose, zurueckgehaltene Frames, Faltungsablehnung, Linien- und
Schachbrettdetektion aus synthetischen Bildern, Vorzeichen, inverse Maps,
Float32-Interpolationsraender und numerisch unveraenderten ZIP-Roundtrip.

Zusaetzlich im integrierten Browser mit Playwright geprueft: ein echtes,
lokal erzeugtes H.264-MP4 mit 640 x 480 Pixeln und bewegtem Linienraster,
Vorwaerts-/Rueckwaertsschritt, Phase-1-Freigabe, Start/Pause/Fortsetzen,
Einzelframe ohne doppelte Beobachtung, Export/Import ohne Video, Wiederzuordnung
des Videos, Deaktivierung mit Neufit und Darstellung bei 390 px Fensterbreite.
Diese synthetischen Tests belegen keine Genauigkeit fuer eine reale Optik.

## Bekannte Grenzen

- Reale Mikroskopaufnahmen wurden noch nicht als unabhaengige Abnahme verwendet.
  Vor produktiver metrischer Nutzung ein eigenes Kalibriervideo und einen
  separaten Kontrolllauf mit bekanntem Target pruefen.
- Der Liniendetektor verfolgt dunkle Linienmittelpunkte und erlaubt gekruemmte
  Verlaeufe. Er setzt lokal getrennte, ausreichend kontrastreiche Linien und
  zwei naeherungsweise quer zueinander liegende Richtungen voraus. Extreme
  Perspektive, enge Radien, unterbrochene/helle Linien, Text und unscharfe
  Kreuzungen koennen zur Ablehnung oder zu Fehlzuordnungen fuehren. Mindestens
  drei Reihen, drei Spalten und neun brauchbare Kreuzungen sind erforderlich.
- Der eigene Schachbrettdetektor ist kein OpenCV-SB-Ersatz fuer beliebige
  Perspektiven und Verdeckungen. Teilmuster sind nur bei rekonstruierbarer
  lokaler Topologie brauchbar. Rechteckige Zellen werden nicht unterstuetzt.
- Die Splineweite wird manuell gewaehlt. Automatische validierungsgetriebene
  Gitterverfeinerung und gemeinsame nichtlineare Nachoptimierung sind nicht
  implementiert. Unterschiedliche Gitterweiten/Glattheiten mit unabhaengiger
  Validierung vergleichen; keine universelle Gewichtung annehmen.
- Positive Jacobians und diskrete Ueberlappungspruefungen sind numerische
  Kriterien, kein analytischer Beweis globaler Invertierbarkeit. Die
  Abdeckungsmaske ist eine grobe raeumliche Stichprobenkarte. Das Qualitaetslabel
  `validated` bezeichnet die eingebauten numerischen Kriterien, keine
  metrologische Zertifizierung.
- MP4-Unterstuetzung haengt vom installierten Edge-Codec ab. Verschluesselte
  Spuren, wechselnde Samplekonfigurationen, komplexe Editlisten,
  Rotations-/Spiegelungsmatrizen und abweichende Anzeigecrops werden abgelehnt.
  Rueckwaertsschritte ueber lange GOPs koennen langsam sein. Nur die erste
  Videospur wird verarbeitet; Audio gehoert nicht zur Kalibrierung.
- Dichte Maps werden vollstaendig gespeichert: 768 MiB Arraybudget bei der
  Erzeugung, 768 MiB entpacktes Importlimit und hoechstens 32 MiB JSON-Metadaten.
  UI, ZIP, Vorschau und zeitweise alte/neue Versionen benoetigen zusaetzlichen
  Speicher. Dies ist kein Gigapixel-Map- oder Mosaikeditor. Feldvisualisierung
  im Hauptthread kann bei sehr grossen Einzelbildern kurzzeitig bremsen.
- Optik, Zoom, Fokus, Crop, Pixelorientierung und Objektebene muessen konstant
  bleiben. Welliges Papier, Druckfehler, Hoehenaenderung oder Verkippung werden
  nicht durch mehr Frames geheilt. Pixelaufloesung ist keine Messgenauigkeit.