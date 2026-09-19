# Rasterlabor

Lokales Browserwerkzeug zur Mikroskopkalibrierung aus einem bewegten ebenen
Checkerboard. Es berechnet ein gemeinsames kubisches
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

## Referenzmuster

`npm run pattern` erzeugt `checkerboard-100x100-1.016mm-300dpi.pdf`: eine
randlose, vektorbasierte Seite von exakt 101,600 x 101,600 mm mit 100 x 100
Schachbrettfeldern. Jedes Feld ist 1,016 mm breit, also exakt 12 Druckpixel bei
300 dpi. Damit fallen an den Schwarz-Weiss-Kanten keine Teilpixel an. Beim
Drucken unbedingt **Tatsaechliche Groesse / 100 %** waehlen und jede automatische
Anpassung an Papier oder Druckbereich ausschalten. Die Seite ist kleiner als A4
und kann zentriert gedruckt werden. Ein anderer Dateiname ist mit
`npm run pattern -- mein-muster.pdf` moeglich.

## Schachbrettflaechen im entzerrten Bild

In der Ansicht **Entzerrt** analysiert **Flaechen analysieren** das pausierte
Bild in nativer Ausgabeaufloesung. Die erwartete **Feldkante, px** wird aus der
Kalibrierung vorbelegt und kann angepasst werden. Ueberlappende Suchbereiche
erfassen vollstaendige Schachbrettfelder; doppelte Treffer und Felder an
ungueltigen Bildbereichen werden verworfen. Querprofile suchen die tatsaechlichen
Hell-Dunkel-Uebergaenge um jede geschaetzte Polygonseite. Eine robuste Geradenanpassung
vertraegt einzelne gestoerte Profile und korrigiert versetzte Ecken durch die
Schnittpunkte der gemessenen Kanten. Mindestens acht von dreizehn Profilen pro
Seite muessen die jeweilige Kante stuetzen. Weder rechte Winkel noch gleiche
Flaechen werden erzwungen. Nicht ausreichend belegte Felder werden ausgeschlossen.
Nicht jedes sichtbare Feld wird
zwangslaeufig erkannt, insbesondere an Raendern oder bei fehlenden Ecken.

Die vier subpixelgenauen Ecken bestimmen die Polygonflaeche in Ausgabepixeln
zum Quadrat. Jedes Feld erhaelt seinen Flaechenwert und eine transparente Farbe.
Die Statistik nennt Anzahl, Minimum, Maximum, Mittelwert, Median und
Standardabweichung. Die relative Abweichung ist `(Maximum - Flaeche) / Maximum`.
Gruen steht fuer die groesste Flaeche, Rot fuer die groesste gemessene Abweichung;
die Farbskala wird auf diesen beobachteten Bereich gespreizt. Ihre Legende zeigt
die tatsaechlichen Prozentwerte. Zwischen Bildern daher auch die Zahlen vergleichen.

Die Analyse laeuft abbrechbar in einem eigenen Worker. Die Ueberlagerung ist
ausblendbar; Frame- und Kalibrierungswechsel verwerfen alte Ergebnisse.
Dies ist ausschliesslich eine Diagnose, keine nachtraegliche Skalierung oder
Aenderung der Entzerrung beziehungsweise des Trackings.

## Ablauf

Window-Tracking registriert das ausgewaehlte entzerrte Fenster gegen den
unmittelbar vorherigen Frame und akkumuliert X/Y/Rotation. Die experimentelle
Loop-Korrektur inklusive Archiv und Diagnose ist entfernt.

## Helligkeitskorrektur

Die Arbeitsflaeche `Helligkeitskorrektur` schaetzt ein ortsfestes Feld nach der
geometrischen Entzerrung. `Von Frame` und `Bis Frame` legen den verwendeten
Bereich einschliesslich beider Grenzen fest. Ein leeres Endfeld bedeutet bis
zum letzten akzeptierten Checkerboard-Frame. Aus den gespeicherten
Kalibrierungsbeobachtungen wird eine zeitlich gleichmaessige Auswahl bis zur
eingestellten Zahl `Ziel-Frames` gebildet. Mindestens fuenf Frames werden fuer
Training und Validierung benoetigt.
`Starten` beginnt eine neue Sitzung. `Pause` stoppt nach dem laufenden
Frame; bereits extrahierte Blockstatistiken bleiben erhalten. Nach einer
Aenderung des Framebereichs nimmt `Fortsetzen` weitere Boardpositionen in die
Sitzung auf. `Zuruecksetzen` verwirft diese Zwischenstaende. Ziel-Frames und
maximale Verstaerkung sind waehrend einer bestehenden Sitzung gesperrt.
Die erkannte Checkerboard-Geometrie wird in das entzerrte Bild abgebildet. Pro
Frame bestimmt der Helligkeitsunterschied der beiden Paritaeten, welche Felder
weiss sind. Nur um 24 Prozent eingerueckte Innenflaechen dieser weissen
Papierfelder werden in 8-mal-8-Pixel-Bloecken gemessen. Schwarze Felder,
Feldkanten, transparente oder gesaettigte Pixel und lokal stark streuende
Bloecke gehen nicht in den Fit ein. PCB-Inhalt und PCB-Tracking werden nicht
verwendet. Die Helligkeitsanalyse arbeitet ausschliesslich aus den weissen
Checkerboard-Papierfeldern; einen PCB- oder allgemeinen Bildueberlappungs-Fit
gibt es nicht.

Im linearen Farbraum wird abwechselnd ein Belichtungs-Offset je Frame und das
sensorfeste lokale Log-Helligkeitsfeld geschaetzt. Eine schwache
Nachbarschaftsregularisierung entfernt Blockrauschen, erhaelt aber lokale weiche
Ringe und Flecken. Das Feld wird bilinear als Float32-Gain in voller entzerrter
Aufloesung ausgewertet. Nur Bereiche mit Messungen aus mindestens zwei
Trainingsframes werden angewendet; ungestuetzte Bereiche behalten Gain 1.

Die globale Option `WebGPU-Beschleunigung` im Kopfmenue ist standardmaessig
aktiviert und gilt gemeinsam fuer Kalibrierung, Tracking, Vorschau/Overlay und
Helligkeitsanwendung. Der Helligkeitsfit auf den weissen Messbloecken selbst
laeuft auf der CPU. Videodekodierung und Entzerrung bestimmten in den bisherigen
Realtests den groessten Teil der Gesamtlaufzeit.

Komplette Frames bleiben zur Validierung zurueck. Das Feld wird nur aktiviert,
wenn deren weisse Innenflaechen nach der Korrektur ausreichend konsistenter sind.
Die Validierung nutzt unabhaengige Frames und die tatsaechlich begrenzten
Gain-Werte. Geringe Stuetzung allein verhindert die Aktivierung nicht.
Die glatte Modellflaeche wird innerhalb der Bilddatenmaske ausgewertet. Die
maximale Verstaerkung gilt weiterhin.
Die einzige Bildansicht zeigt die
relative Durchlaessigkeit `V(x)` mit einer robusten automatischen, um 0 Prozent
symmetrischen Grauskala und unverzerrtem Seitenverhaeltnis. Ungemessene und
maskierte Bereiche sind transparent. Alte Spline-Feldpakete bleiben lesbar
und behalten ihr bisheriges Verhalten; neue Berechnungen sind pixelweise.

Die begrenzte Verstarkung wird nach jedem entzerrten Frame auf CPU und WebGPU
in linearem Licht angewendet, einschliesslich Window-Tracking, Inspektion,
Ueberlagerung und Export. Das `.rbright`-Paket speichert Float32-Gain,
Stuetzmaske, Solver-/Validierungsdaten, Normalisierung und Farbraumannahme. Ein
SHA-256-Fingerprint bindet es an Quell- und Zielgeometrie,
Entzerrungskoeffizienten, Crop-Origin und Optikbezeichnung; Abweichungen werden
beim Laden abgelehnt.
Beim Speichern der geometrischen Kalibrierung wird das aktive Helligkeitsfeld
zusaetzlich in das ZIP eingebettet und beim Import automatisch aktiviert. Ein
Paket ohne eingebettetes Feld deaktiviert eine zuvor aktive Korrektur. Der
separate `.rbright`-Import und -Export bleibt fuer den gezielten Austausch
zwischen passenden Geometrien erhalten.

Die Methode setzt gleichmaessig weisses Papier und verlaesslich erkannte
Checkerboard-Zellen voraus. Bewegte Schatten, Reflexe und automatische
Belichtung sind keine statischen optischen Fehler und koennen das Feld
verfaelschen. Die Validierung auf zurueckgehaltenen Checkerboard-Frames prueft
deshalb, ob die weissen Innenflaechen nach der Korrektur konsistenter sind.

## Merge-Grossbild

Die Arbeitsflaeche `Merge` setzt alle gueltigen Tracking-Posen bei nativer
entzerrter Aufloesung zu einem Bild zusammen. Linsenfeld, Tracking-Maske,
Randueberblendung und eine aktive Helligkeitskorrektur werden genauso wie in der
Tracking-Ueberlagerung angewendet. Die Frames laufen deterministisch von
scharf nach unscharf. Der GPU-Pass nimmt je Ausgabepixel nur die ersten `n` und
damit schaerfsten gueltigen Beitraege an. Wahlweise werden diese gewichtet
gemittelt oder in umgekehrter Alpha-Richtung zusammengesetzt; das Ergebnis ist
dabei identisch zu unscharf nach scharf, ohne alle Frames ein zweites Mal zu
dekodieren.

Die Ausgabe wird kachelweise auf der GPU berechnet und unmittelbar als
unkomprimiertes RGBA-BigTIFF geschrieben. Weder ein Canvas noch ein RGBA-Puffer
in Gesamtbildgroesse wird angelegt; dadurch sind auch Abmessungen oberhalb des
GPU-Texturlimits und Dateien groesser als 4 GiB moeglich. Nicht belegte
BigTIFF-Kacheln bleiben sparse. Abbrechen verwirft die unvollstaendige Datei.

Ein inkrementeller Matchverlust wird weiterhin angehalten, nicht verdeckt.

Die Bilddatenmaske im Videooptionen-Dialog wird auf einem orientierten Rohframe
gemalt und benoetigt deshalb keine geladenen Korrekturdaten. Patch-Tracking nutzt
sie direkt. Fuer Window-Tracking und Ueberlagerungen wird sie bei vorhandener
Kalibrierung ueber deren inverse Map in entzerrte Koordinaten projiziert.

Umfeldstabilisierung registriert zusaetzlich gegen bis zu n letzte und m scharfe,
raeumlich verteilte Referenzen (Standard jeweils 2, Bereich 0 bis 8; beide 0
deaktiviert die Zusatzregistrierung). Alle bisherigen Frames des aktuellen Laufs
bleiben mit Pose und 16x16 Scharfe-/Gueltigkeitsraster im Kandidatenpool. Scharfe
wird innerhalb der vorhergesagten maskierten Schnittmenge bewertet; die Auswahl
bevorzugt zusaetzlich noch nicht abgedeckte Teile des aktuellen Bildes und
getrennte Referenzpositionen. Unter 16 gemeinsamen Rasterzellen wird verworfen.
Ein automatisch dimensionierter LRU-Bildcache beschraenkt nur die gehaltenen
Grauwertpyramiden; fehlende Referenzen werden gezielt dekodiert und entzerrt.
Das Budget ist Pyramidengroesse mal (1 + n + m), mindestens 192 und hoechstens
768 MiB. Bei 90 MiB je Pyramide und n=2/m=4 sind es 630 MiB fuer sieben Bilder.
Die Obergrenze kann weiterhin Verdraengungen verursachen. Explizite Budgets im
ContextTracker-Konstruktor bleiben fest. Reset gibt Cache und GPU-Ressourcen
frei. Aktuelle Pyramide, Maps, GPU-Bilder und kurzlebige RGBA-/Decoderpuffer
benoetigen zusaetzlichen Speicher; das Cachebudget ist kein Gesamtspeicherlimit.

Die Registrierung nutzt ausschliesslich gueltige Pixel der vorhergesagten
Schnittmenge, mit begrenzter X/Y-/Rotationsnachsuche (Standard 32 px und 1 Grad).
Sie arbeitet grob nach fein bis zu nativen Pixelwerten, mit maximal etwa 6000
raeumlich verteilten Messpunkten pro Pyramidenstufe und bilinearer Interpolation.
Der Radius begrenzt die lokale iterative Verfeinerung, keine erschoepfende Suche
aller Lagen. Groessere Vorhersagefehler koennen ausserhalb des Konvergenzbereichs
liegen und trotz vorhandener Ueberlappung abgelehnt werden.
NCC >= 0.9, lokale Eindeutigkeit und Abstand zur Suchgrenze sind erforderlich.
Die Vorwaerts-/Rueckwaertspruefung ist bis 1.5 px streng gueltig. Raeumliche
Paare bis standardmaessig 7.5 px bleiben bedingte Kandidaten; aktuelle Referenzen
werden dadurch nicht gelockert. Beide Grenzen sind in den Trackingoptionen
einstellbar. Der Inspector zerlegt den Zyklusfehler in Translation, Rotation
sowie Median, RMS und P95 des Bildpunktversatzes auf den gueltigen maskierten
Ueberlappungszellen.

Mindestens zwei Messungen und eine strikte Mehrheit muessen eine gemeinsame Pose
stuetzen. Zeitlich benachbarte Referenzen zaehlen dabei als eine Referenzgruppe.
Bedingte raeumliche Kandidaten duerfen eine Loop-Closure nur ausloesen, wenn
entweder zwei getrennte Referenzgruppen zustimmen oder derselbe Korrekturvorschlag
in zwei aufeinanderfolgenden aktuellen Frames gegen dieselbe Referenzregion
wiederkehrt. Der robuste Konsens korrigiert die aktuelle Pose, verteilt die
Closure ab dem Referenzanker auf den bisherigen Pfad und setzt die korrigierte
Pose als Basis des naechsten Schritts. Ohne bestaetigten Konsens bleibt die
inkrementelle Pose erhalten.

Nach Auswahl eines Punktes im XY-Pfad kann `Pfad nachoptimieren` die vorhandenen
Frames aus zeitlich getrennten Besuchen dieser Flaeche lokal neu gegeneinander
registrieren. Vor der Einzelbildsuche werden die zeitlich zusammenhaengenden Besuche
getrennt zu lokalen Mischbildern aufgebaut. Auswahl der scharfen Bilder,
Welttransformation, Pixelmaske, Kantenuebergang und Mittelung entsprechen der
Mischbildanzeige; der Ausschnitt bleibt auf den gewaehlten lokalen Bereich begrenzt.
Alle Gruppenpaare werden vorwaerts und rueckwaerts registriert. Bestaetigte
Gruppenkorrekturen werden vom Besuch des ausgewaehlten Frames durch das verbundene
Gruppennetz weitergegeben. Sie dienen ausschliesslich als Startlage fuer das
anschliessende Einzelbild-Refinement und erzeugen selbst keine Loop-Closure-Kante.
Mehrere scharfe Einzelbilder werden danach kreuzweise vorwaerts und rueckwaerts
geprueft. Kandidaten brauchen mindestens 128 nutzbare Samples innerhalb der lokalen
Maske. Die Suche arbeitet adaptiv mit bis zu 64 neuen Paaren pro Runde und insgesamt
512 Paarversuchen. Ohne bestaetigten Match werden weitere Vertreter nachgezogen;
Vorwaertstreffer ab NCC 0.9 werden mit benachbarten Frames beider Besuche geprueft.
Hoechstens die Haelfte einer Runde wird fuer diese Nachbarpruefungen reserviert,
damit die weitere Suche nicht stehen bleibt. Besuchspaare werden abwechselnd
bedient, bereits gepruefte Paare nicht wiederholt. Pro Runde werden hoechstens
32 Vollbilder geladen und danach freigegeben. Alle Rundenergebnisse fliessen in
den gemeinsamen Konsens ein. Die Suche stoppt bei bestaetigten Graphkanten,
ausgeschoepften Kandidaten oder erreichtem Suchbudget; der Status unterscheidet
diese Faelle. Nach bestaetigten Graphkanten wird die Korrektur ueber die verbundene
Bildkomponente verteilt. Mit den korrigierten Posen werden Kandidaten und Masken
neu berechnet und weitere lokale Kanten gesucht. Bis zu drei solcher Refit-Iterationen
werden ausgefuehrt; eine Iteration ohne neue bestaetigte Kante beendet den Vorgang.
Die lokalen Suchgrenzen bleiben dabei unveraendert, und der gesamte Ablauf bleibt
bis `Uebernehmen` eine Vorschau. Die lokale Registrierung erweitert ihren Radius iterativ bis zur
Bilddiagonale und uebernimmt gute Randtreffer als Startpunkt der naechsten Stufe.
Eine konsistente Mehrheit aus mindestens zwei verschiedenen Frames auf beiden
Seiten erzeugt neue lokale Graphkanten. Alternativ darf ein einzelnes regulaer
beidseitig akzeptiertes Paar mit NCC >= 0.98, Marge >= 0.002, mindestens 128 Samples
je Richtung und Zyklusfehler <= 1.5 px eine Kante bilden. Die Grenzen werden bei
weiteren Suchrunden nicht gelockert. Fuer Mehrpaar-Konsens liegt die Zyklus- und
Korrekturgrenze je nach eingestellter bedingter Zyklusgrenze zwischen 25 und 50 px. Eine
robuste Gewichtung begrenzt widerspruechliche Loop-Kanten. Der rot gestrichelte
Pfad ist nur eine Vorschau und veraendert die Trackingdaten erst mit `Uebernehmen`;
`Verwerfen` entfernt ihn. Der Refit optimiert X, Y und Rotation, aber keinen Massstab.

Ein inkrementeller Matchverlust wird weiterhin angehalten, nicht verdeckt.
Ein verworfener Window-Match laesst den letzten gueltigen Frame und die Pose
unveraendert und pausiert vor dem fehlgeschlagenen Frame. Suchradius,
Rotationsgrenze und Umfeld-Suchgrenzen koennen danach angepasst werden;
**Fortsetzen** wertet denselben Frame erneut aus. Der fehlgeschlagene Versuch
wird weder in den Pfad aufgenommen noch als verarbeitet markiert. Die nutzbare
Translationsgrenze ist der kleinere Wert aus Suchradius und halber Fensterbreite
bzw. -hoehe; Verschiebungen ab einer halben Fensterdimension sind durch die
periodische FFT-Translation nicht eindeutig. Ausnahmen wie Worker-/Decoderfehler
bleiben ein nicht fortsetzbarer Abbruch.

**WebGPU mit CPU-Fallback:** Die vorhandene WebGPU-Option gilt auch fuer das
Umfeldtracking. Maskierte Grauwertpyramiden werden auf der GPU aufgebaut; die
CPU erhaelt einmalig eine Kopie fuer Referenzauswahl, Schnittmengenraster und
Fallback. Beim Window-Tracking mit aktiver Umfeldstabilisierung werden native
VideoFrames an den Tracking-Worker uebergeben. Er enthaelt eine Kopie der Maps
pro Kalibrierung. Entzerrung und Pyramidenaufbau verwenden dasselbe GPU-Geraet;
die erste Pyramidenstufe liest direkt aus der entzerrten Textur. Referenzen
benoetigen weder einen vollaufgeloesten RGBA-Readback noch erneuten RGBA-Upload.
Nur das aktuelle Bild wird auch als RGBA fuer das unveraenderte CPU-Fenstertracking
gelesen; die Hauptseite erhaelt dessen Vorschau als ImageBitmap. Diese RGBA-Daten
bleiben nicht im Pyramidencache. Die Grauwertkopien aller Stufen bleiben fuer
CPU-Paarregistrierung und Referenzauswahl erforderlich. Native GPU-Fehler fallen
mit demselben VideoFrame auf CPU-Entzerrung/Pyramiden zurueck. Uebergebene Frames
werden im Worker auch bei Fehlern geschlossen. Beide Backends verwenden dieselbe
ganzzahlige Grauwertquantisierung.
Fuer Paarmessungen kann die GPU bilineare Bildabtastung, Korrelationssummen und
robuste Normalgleichungen berechnen. Pro getesteter Pose kommen nur 96 Byte
Summen zurueck; Bildpuffer bleiben zwischen den Iterationen resident. Das kleine
3x3-System, Suchsteuerung und Konsens bleiben auf der CPU. Der synchrone CPU-Pfad
und der asynchrone GPU-Pfad teilen Such- und Akzeptanzlogik.

Die erste Paarmessung eines Laufs wird mit beiden Backends gemessen. Nur bei
gleicher Akzeptanz, maximal 0.1 px Posenabweichung einschliesslich Winkelversatz
am Bildrand und kuerzerer gemessener Laufzeit wird die GPU-Paarregistrierung
beibehalten. Andernfalls werden GPU-Pyramiden mit CPU-Paarmessungen kombiniert;
der Diagnosegrund und beide Probezeiten stehen in `context.registrationChoice`.
Diese einmalige Probe erhoeht die Anlaufzeit und ist keine Garantie fuer jede
spaetere Bildsituation. Pufferlimits, fehlendes WebGPU, Shaderfehler und
Geraeteverlust wiederholen die betroffene Operation auf der CPU und deaktivieren
den defekten GPU-Backend bis zum Tracking-Neustart. Ablehnungen mangels Struktur
oder Konsens bleiben normale Ablehnungen, keine erzwungenen CPU-Neuversuche.
Mit ausgeschalteter WebGPU-Option bleibt alles auf der CPU.

Der GPU-LRU-Cache hat zusaetzlich maximal 192 MiB Nutzlast, neben dem CPU-Bildcache
und temporaeren Upload-/Readbackpuffern. Er haelt keine zusaetzlichen CPU-Bilder
fest. Bei CPU-Paarmessungen werden residente GPU-Bilder freigegeben; GPU-Pyramiden
verwenden dann nur temporaere Puffer. Referenzdekodierung und Bildcache-Auswahl
werden durch die GPU-Option nicht geaendert.

Die Match-Anzeige nennt Konsens/Referenzzahl, im Tooltip stehen Frame-IDs und
Ablehnungsgruende. tracking.json enthaelt pro Frame `incremental` und `context`
einschliesslich Einzelmessungen. Backend und CPU-Fallback stehen in der
Match-Anzeige; der Tooltip nennt den Grund und die Bildcache-Treffer/Nachladungen.
Ein Klick auf eine Zeile der Posenliste oeffnet darunter die Matchdiagnose:
Referenzversuche, unveraenderte historische Start-/Referenzposen, jede Suchstufe
und die separate Rueckwaertspruefung. Die Paaransicht zeigt Ueberlagerung oder
Differenz, Laufmaske beziehungsweise Fenstergrenze sowie Zoom und Verschiebung.
Debugversuche verwenden einen eigenen CPU-Worker und die gespeicherte Laufmaske;
Radius, Winkel, Grobsuche und Rueckradius sind einstellbar. Sie pruefen die
Umfeldregistrierung auch bei einem inkrementellen Referenzpaar, nicht den
FFT-Fenstertracker. Sie veraendern keine Pfadpose. `context.matches[].debugAttempts`
speichert diese Versuche neben dem Original; `incrementalMatch` dokumentiert den
Fenstervorgaenger. Verworfene Fensterframes bleiben getrennt unter `failures`
erhalten und erscheinen ebenfalls in der Liste. Alte Datensaetze ohne Startposen
erhalten keine erfundene historische Ausrichtung.
Rueckwaertspruefung ist eine Paarvalidierung. `loopClosure` dokumentiert dagegen
die rueckverteilte Pfadkorrektur; dies ist noch keine iterative Posegraph-Optimierung.
Das Zeitprofil trennt Referenzladen/Entzerren, Pyramidenaufbau inklusive Transfer
und Paarregistrierung vorwaerts/rueckwaerts. Im direkten GPU-Pfad beinhaltet die
Pyramidenzeit auch native Entzerrung; Referenzladen besteht dann nur aus dem
Frameabruf. `nativeImages` zaehlt direkte GPU-Bilder, `rgbaReadbackBytes` die
zusaetzliche Fenster-/Vorschaukopie. `uploadBytes` misst explizite Buffer-Uploads,
nicht Map-Textur-Uploads oder interne Decoder-/Treibertransfers. Diese sind Teilzeiten, nicht
zusaetzlich zu "Umfeld gesamt" zu addieren. Pausieren beendet den laufenden Workerauftrag und
ueberspringt noch nicht angeforderte Referenzen. Alte importierte Pfade werden
nicht automatisch als Referenzen uebernommen. Validierung zuerst an kurzen
Ausschnitten, nicht am Vollvideo; hohe Paar-NCC beweist keine driftfreie Karte.

GPU-Regressionstest im Browser (separate leere Seite, kein Videolauf):
`npx esbuild tests/context-gpu.browser.js --bundle --format=esm --outfile=dist/context-gpu-check.js`
nach `npm run build` ausfuehren. Auf dem lokalen Server dann
`await (await import('/context-gpu-check.js')).runContextGpuChecks()` aufrufen.
Der Test prueft Shader, pixelgleiche Pyramiden, Maskenausschluesse, X/Y/Rotation,
Konsens, Pufferlimit-Fallback, Geraeteverlust und Reset. Optional misst
`benchmarkContextGpu(4096, 3072)` ein einzelnes synthetisches Bildpaar; keine
Aussage ueber die End-to-End-Laufzeit eines realen Videos.

### Ueberlagerung und Detailprofil

Die Ueberlagerung beginnt mit allen Frames, deren aus der Tracking-Pose
approximierte Bildflaeche den angeklickten Punkt abdeckt. `Frames je Bereich`
erlaubt 1 bis 64. Diese Kandidaten werden nach gespeichertem
Ganzbild-Schaerfewert sortiert und belegen danach ein grobes Weltkoordinatenraster.
Seine Zellkante entspricht 10 Prozent der entzerrten Framebreite und damit der
Randueberblendung; kleinere Positionsunterschiede erzwingen keinen weiteren Frame.
Ein Frame wird vor der Dekodierung verworfen, wenn jede von ihm abgedeckte Zelle
bereits n schaerfere Frames besitzt. Damit bleiben raeumlich weiterreichende
Frames fuer das Mosaik erhalten, waehrend nahezu deckungsgleiche Wiederholungen
nicht geladen werden. Es wird keine lokale Schaerfekarte neu berechnet. Fehlende
Werte kommen zuletzt; Gleichstand entscheidet die aufsteigende Frame-ID. CPU,
GPU und Kacheln verwenden dieselbe vorausgewaehlte Liste. Der pixelweise Zaehler
bleibt als gewichtete Sicherheitsgrenze bestehen: Teilgewichtete Feather-Pixel
lassen Restgewicht fuer den naechsten Frame, sodass auch `Top 1` weich mischt.
Bei Kacheln kann ein ausgewaehlter Frame mehrfach durchlaufen. Die Platzierung
verwendet dieselbe stabilisierte Pose wie Pfad und Export, nicht die rohe Pose.
Jeder entzerrte Frame wird an allen vier Bildraendern weich ausgeblendet. Die
Rampentiefe bezieht sich auf die entzerrte Bildbreite und endet spaetestens in
der Bildmitte, sodass auch bei 50 Prozent ein voll gewichteter Kern verbleibt.
Sie verwendet Smoothstep statt einer linearen Kante. Der Faktor gewichtet Farbe
und Alpha vor der Mittelung identisch. Die Gewichte steuern nur die Farbmischung:
Jeder von mindestens einem Frame belegte Ausgabepixel bleibt deckend, damit an
Bild- und Maskenraendern kein weisser Hintergrund durchscheint. Vollstaendig
unbelegte Pixel bleiben transparent. Maskierte Pixel verbrauchen keinen
Top-n-Platz, jeder sonstige teilweise gewichtete Frame einen Platz.
Die Kachelgroesse wird aus GPU-Texturlimit und verbleibendem 1,5-GiB-Bildbudget
als Zweierpotenz gewaehlt, maximal 4096 px. Damit ersetzen ueblicherweise
4096er-Kacheln das fruehere feste 2048er-Raster. Eine 5x6-Aufteilung sinkt bei
gleichen Mosaikabmessungen typischerweise auf etwa 3x3; groessere Kacheln
enthalten mehr Kandidaten, aber jeder Kandidat wird in weniger Kacheln erneut
dekodiert. Fortschritt und Ergebnis nennen Kachelkante und gesamte
Frame-Durchlaeufe. Die Vorauswahl ist bewusst eine schnelle Approximation aus
den bekannten Posen; die eigentliche Mischung bleibt pixelgenau.
Die GPU-Ueberlagerungsanzeige trennt Frameabruf und GPU-Verarbeitung inklusive
Entzerrung/Synchronisation; Kacheln melden auch Durchlaeufe und Ausgabezeit.

Das Tracking-Zeitprofil trennt CPU-Pyramiden, GPU-Initialisierung, Uploadkopie,
Puffer/Befehle, mapAsync-Wartezeit und CPU-Rueckkopie. Optional verfuegbare
GPU-Zeitstempel messen ausschliesslich die Pyramiden-Kernel. Diese Zeit liegt
innerhalb der Wartezeit, nicht zusaetzlich dazu; Wartezeit ist keine reine
Transfermessung. Referenzabrufe zeigen Dekodierung, Schaerfe, RGBA und Entzerrung.
`context.pyramidProfile` speichert Zeiten, Bildzahlen und Transferbytes;
Cachebelegung, Budget, Kapazitaet und Verdraengungen stehen im Match-Tooltip.
Cache-Misses zaehlen angeforderte Referenzen, auch wenn Pause oder Fehler deren
Laden spaeter verhindern. Das Profil zeigt Mittelwerte des Laufs, der Tooltip
den einzelnen Frame. Importierte Daten ohne diese Felder bleiben lesbar.

Der Umfeld-GPU-Backend fordert nun bis zu 256 MiB Storage-Bindings an, begrenzt
durch den Adapter, statt nur das WebGPU-Standardlimit zu verwenden. Groessere
Bilder melden angeforderte MiB und Limit im CPU-Fallback-Grund.

Kurze synthetische Messung am 15.09.2026 auf dem vorhandenen Adapter, jeweils
zwei Pyramiden, keine reale Aufnahme und kein End-to-End-Benchmark:

| Bildgroesse | CPU | GPU gesamt | GPU-Kernel | Pyramide | Cachekapazitaet (192 MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 x 3072 | 927 ms | 135 ms | 0.78 ms | 32 MiB | 6 |
| 8192 x 4320 | 2404 ms | 330 ms | 4.44 ms | 90 MiB | 2 |

Beim grossen Paar entfallen rund 114 ms auf Uploadkopien, 173 ms auf Warten und
39 ms auf CPU-Rueckkopien. Ein reiner Cachezugriff mit LRU-Aktualisierung liegt
unter 0.001 ms, enthaelt aber weder Registrierung noch Dekodierung. Einzelproben
schwanken durch JIT, Garbage Collection und andere GPU-Nutzer.
Cache-Dimensionierung und direkte GPU-Weitergabe sind inzwischen implementiert.
Die Tabelle oben dokumentiert den vorherigen RGBA-Pfad. Ein nachfolgender
Vergleich mit `benchmarkNativeContext(width, height)` misst drei identische
Referenzbilder je Pfad, ohne Dekodierung oder Paarregistrierung; alle
Pyramidenpixel werden verglichen. Mediane auf demselben Adapter:

| Bildgroesse | RGBA-Umweg inkl. Entzerrung | Direkt GPU | Eingespart je Referenz |
| --- | ---: | ---: | --- |
| 4096 x 3072 | 182 ms | 26 ms | 48 MiB RGBA hin + 48 MiB zurueck |
| 8192 x 4320 | 441 ms | 118 ms | 135 MiB RGBA hin + 135 MiB zurueck |

Die grosse Probe schwankte zwischen 321-942 ms vorher und 75-258 ms direkt.
Anlauf, Garbage Collection und GPU-Last beeinflussen Einzelmessungen; keine
Hochrechnung auf die reale Gesamtframerate. Die native Map-Initialisierung ist
im direkten Benchmark vorgewaermt. Das GPU-Geraet fordert fuer interne grosse
Map-Upload-Stagingpuffer das vom Adapter unterstuetzte maxBufferSize an; dies
reserviert nicht automatisch Speicher und aendert das Cachebudget nicht.
`runNativeContextChecks()` prueft Masken, gedrehte/gespiegelte Frames, pixelgleiche
Pyramiden, Referenznachladen, Geraeteverlust-Fallback sowie den echten
Worker-Protokollpfad mit Vorschau, Cachetreffern und Konsens.
Weitere moegliche Optimierungen, noch nicht implementiert: Wiederverwendung
temporaerer Puffer und weniger Frame-/Kachel-Wiederholungen in der Ueberlagerung.

1. Video oeffnen. Es wird noch nichts kalibriert. Aufloesung, Codec,
  Praesentationszeitstempel und Frameindex erscheinen nach der Indexierung.
  Play/Pause, Zeitleiste und die Einzelbildtasten sind davon
   unabhaengig. Einzelbildschritte dekodieren ab dem vorausgehenden Keyframe.
2. **Erkennung pruefen**. Optional Rasterabstand, Kontrast und
   Erkennungsbereich anpassen. Spalten/Zeilen sind innere Schachbrettecken,
   nicht Zellen. Die automatische
   Abstandsschaetzung setzt den Startwert fuer Zielpixel pro Rasterweite.
   Bei der Kalibrierung bleibt dieser Massstab fest.
3. Overlay an mehreren Stellen kontrollieren: erkannte innere Ecken,
   zusammenhaengende Indizes und keine ausgelassenen Ecken. Die gruene
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

- Die XYR-Frame-Ueberlagerung verwendet bei aktiviertem WebGPU native
  `VideoFrame`-Texturen. Rotation, vollaufloesende Linsenentzerrung, Maske,
  XYR-Ausrichtung und Float32-Mittelwertbildung laufen auf der GPU, ohne
  RGBA-Readback pro Frame. Das fertige Bild bleibt in voller Pixelaufloesung;
  Einpassen und Zoom aendern nur die Anzeige. Ueberschreitet das gesamte Bild
  das GPU-Texturlimit oder das Arbeitspufferbudget, werden automatisch
  2048-x-2048-Kacheln berechnet. Dabei bleiben Maps und nur ein Kachel-
  Summenpuffer aktiv; fertige Kacheln werden vollaufgeloest zur Anzeige
  aufbewahrt. Frames koennen dafuer mehrfach dekodiert werden. Die Maps
  eines einzelnen Quellbildes muessen weiterhin ins GPU-Texturlimit passen.
  Ohne WebGPU
  bleibt der bisherige CPU-/Canvas-Pfad verfuegbar.
- Alle Frame-Verbraucher verwenden `FrameReader`: Rohbildvorschau, Erkennung,
  entzerrte Vorschau, Maskeneditor, Tracking, Ueberlagerung und Videoexport.
  Bei aktiviertem GPU-Modus werden native Decoder-Frames verwendet. Die
  Entzerrung erfolgt vor jedem CPU-Readback; Vorschau und Export erhalten direkt
  ein GPU-Bitmap. Nur Verbraucher von CPU-Pixeln erhalten vollaufgeloeste RGBA-Daten.
  Der Tracking-Worker arbeitet weiterhin auf CPU-RGBA und behaelt seine bisherigen
  Fenster- und FFT-Verfahren. Keine weitere Reduktion der Bildaufloesung.
  GPU-Maps und Readback-Puffer werden wiederverwendet, bei Map-/Videowechsel
  freigegeben; GPU-Ausfaelle schalten auf den CPU-Pfad zurueck.
- Jeder erstmals dekodierte Video-Zeitstempel erhaelt einen Schaerfewert. Dazu
  wird das Bild proportional auf hoechstens 1024 Pixel Kantenlaenge abgetastet
  und die Median-Varianz des Luma-Laplacians in neun Bildregionen berechnet.
  WebGPU fuehrt Abtastung und Momentberechnung aus; bei deaktivierter GPU dient
  Canvas als Rueckfall. Wiederholte Zugriffe verwenden den Wert aus dem Cache.
  Kalibrier-Beobachtungen und Tracking-Eintraege speichern das vollstaendige
  `sharpness`-Objekt in `metadata.json` beziehungsweise `tracking.json`.
  Messungen: [FrameReader / Tracking](benchmarks/FRAME_READER_BENCHMARK.md).
  Messungen und Reproduktion: [GPU-Ueberlagerungsbenchmark](benchmarks/OVERLAY_BENCHMARK.md).

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
