# Fly Brain Lab

Ein Forschungs-Sandbox-Projekt: Das komplette Konnektom einer Fruchtfliege (Drosophila) wird als Spiking-Netz simuliert, steuert einen physikalisch simulierten Fliegenkörper in einer Umwelt, und ein LLM arbeitet als automatisierter Experimentator (Hypothese, Eingriff, Simulation, Auswertung).

> Hinweis an Claude Code: Das hier ist ein eigenständiges, neues Projekt. Baue es für sich allein auf, im eigenen Ordner und eigenen Environment.

---

## 1. Hintergrund

Am 3. September 2026 haben HHMI Janelia und Google Research **MaleCNS v1.0** veröffentlicht, das erste vollständige Konnektom des zentralen Nervensystems (Gehirn + Nervenstrang) einer erwachsenen männlichen Fruchtfliege: rund **166.700 Neuronen** und etwa **125 Millionen Synapsen**. Die Daten sind offen. Seitdem gibt es einen Meme- und Hobby-Trend ("Fly Brain Simulations"): Die Fliege spielt Beat Saber, Minecraft, Doom, tradet Bitcoin oder steuert Drohnen.

**Wichtig:** Ein Konnektom ist ein **statischer Schaltplan**, kein lebendes Gehirn. Es enthält Verbindungen (und teils Vorhersagen zum Neurotransmitter), aber keine gelernten Gewichte, keine Plastizität, keine Neuromodulation. Alles, was daraus "Verhalten" macht, ist ein Modell obendrauf (z. B. Leaky-Integrate-and-Fire). Ergebnisse sind Hinweise, keine Beweise.

---

## 2. Projektziel

Eine Simulationsumgebung, in der:

1. das Fliegen-Konnektom als Gehirn läuft,
2. ein Fliegenkörper in einer Physik-Umgebung mit Luft, einstellbarer Gravitation und Gegenspielern lebt,
3. ein LLM-Agent systematisch Experimente fährt (Neuronen stilllegen, Verbindungen ändern, Szenarien variieren) und auf Basis **berechneter Metriken** auswertet.

### Szenarien

| Szenario | Idee | Metriken |
|---|---|---|
| **Spinne** | Eine Spinne (erst geskriptet, später RL-Policy) jagt die Fliege. Looming-Erkennung und Flucht (Giant-Fiber-Pfad) sind die zu testenden Reflexe. | Überlebenszeit, Fluchtlatenz, Fluchtrichtung |
| **Futtersuche** | Duftquelle oder heller Punkt als Ziel, plus Hindernisse. | Zeit bis Ziel, Pfadeffizienz, Landung ja/nein |
| **Lichtwahl** | Zwei Wege, hell und dunkel. | Anteil Entscheidung pro Seite |
| **Zero-g mit Luft** | Gravitation auf 0 (oder klein), Luft bleibt (Dichte und Viskosität unverändert). Zero-g heißt nicht Vakuum. | Stabilität, Drift, Orientierung, Kontrollaufwand |

### Experimenttypen

- **Lesion:** Neuronentyp oder -gruppe stilllegen.
- **Rewiring:** Verbindungen verstärken, kappen, verschieben.
- **Shuffle-Baseline:** Synaptische Ziele pro Neuron zufällig würfeln (Kontrolle). Verschwindet ein Verhalten dadurch, hängt es an der echten Verdrahtung.
- **Ablation von Sinnesorganen:** z. B. Antennen-Mechanosensoren (Lage und Schwerkraft-Wahrnehmung) stilllegen.

---

## 3. Architektur

```
Umwelt (Physik, Luft, Gravitation, Spinne, Ziele)
   |  Sinnesdaten (Bild, Geruch, Propriozeption)
   v
Sinnes-Frontend (Augenmodell FlyVis, Duft-/Mechano-Input)
   v
Gehirn: Konnektom als Spiking-Netz (LIF, z. B. Brian2)
   v
Descending Neurons -> Motor-Ausleseschicht -> Körper (MuJoCo)
   ^
   |  Experiment-API (Lesion, Rewiring, Szenario, Seed)
LLM-Experimentator (Plan -> Run -> Metriken -> Bericht)
```

**Wichtig:** Die Ausleseschicht (descending neurons zu Gelenk- und Flügel-Aktuatoren) ist meist trainiert und trägt Teile der Arbeit. Das ist ein ehrlicher Teil des Modells und gehört in jeden Bericht ("Konnektom + trainierte Ausleseschicht").

---

## 4. Hardware und Simulator

**Hardware:** AMD RX 590 (8 GB VRAM, Polaris, kein CUDA, kein offizielles ROCm). Brian2CUDA, GeNN und die meisten PyTorch-GPU-Pfade fallen weg.

**Strategie:**

1. Start auf CPU mit einem Teilschaltkreis (Seh- und Fluchtpfad, ein paar tausend Neuronen), ereignisgesteuert mit numba.
2. Eigener LIF-Simulator, Synapsen als CSR-Matrix (Zeiger, Ziel-Index, Gewicht). Pro Zeitschritt nur die gefeuerten Neuronen verarbeiten, danach LIF-Update für alle.
3. GPU-Port über Taichi mit Vulkan-Backend. Fallback: PyOpenCL oder wgpu. Float-Atomics auf Polaris eventuell nicht verfügbar, dann Integer-Fixed-Point mit atomarer Addition.
4. Lesion und Rewiring als Masken auf der CSR-Matrix, ohne Neukompilieren.
5. Früh einen Mini-Benchmark bauen (ca. 10.000 Neuronen, ein paar Mio. Synapsen), Zeitschritte pro Sekunde messen und in `docs/assumptions.md` festhalten.
6. Validierung: Spike-Raten des eigenen Simulators gegen Brian2 und das Shiu-Modell auf demselben Teilschaltkreis vergleichen, bevor ihm vertraut wird.
7. Große Experimentserien (viele Seeds, Lesionen, Baselines) bei Bedarf auf einer gemieteten Nvidia-Cloud-GPU laufen lassen. Entwicklung und Tests lokal.

---

## 5. Ressourcen: Wo bekommt man was

### 5.1 Konnektome (das Fliegenhirn)

| Was | Wo | Notizen |
|---|---|---|
| **MaleCNS v1.0** (neu, 2026, Gehirn + Nervenstrang, ca. 166.700 Neuronen) | Janelia / Google Research Release und **neuPrint** (https://neuprint.janelia.org) | Genauen Download-Link und Datenformat zuerst verifizieren (Release-Seite von Janelia/Google Research prüfen). Vorhandene Community-Repos (siehe 5.4) zeigen, wie sie die Daten laden. |
| **FlyWire** (weibliches Gehirn, v783: 138.639 Neuronen, ca. 54 Mio. Synapsen) | https://flywire.ai und Codex (https://codex.flywire.ai) | Gut dokumentiert, viele fertige Tools bauen darauf auf. Guter Einstieg, falls MaleCNS-Handling hakt. |
| **neuPrint-Python-Client** | `pip install neuprint-python` | Zugriff auf neuPrint-Datensätze per API (Token nötig). |

### 5.2 Gehirnmodelle (Spiking-Simulation)

| Was | Wo |
|---|---|
| **Shiu et al. 2024 LIF-Modell** (FlyWire-Gehirn als Leaky-Integrate-and-Fire, Brian2) | https://github.com/philshiu/Drosophila_brain_model |
| **Brian2** (Simulator) | https://brian2.readthedocs.io, `pip install brian2` |
| **FlyGM** (Konnektom als Graph-Controller, RL) | Paper: https://arxiv.org/html/2602.17997v3 |

### 5.3 Körper, Augen, Physik

| Was | Wo | Notizen |
|---|---|---|
| **flybody** (DeepMind + Janelia, anatomisch detaillierter MuJoCo-Fliegenkörper) | https://github.com/TuragaLab/flybody | Controller dort sind MLPs, kein Konnektom, aber der Körper ist nutzbar. |
| **NeuroMechFly / flygym** (EPFL, biomechanische Fliege, MuJoCo) | https://github.com/NeLy-EPFL/flygym | Laufen, Putzen, Sensorik. |
| **FlyVis** (Fliegenaugen-Modell) | https://github.com/TuragaLab/flyvis | Netzhaut und optische Verarbeitung, Eingang für Kamerabilder. |
| **MuJoCo** | https://mujoco.org, `pip install mujoco` | Gravitation ist Parameter (`model.opt.gravity`), Fluidmodell (Dichte, Viskosität) vorhanden. |

Hinweis: MuJoCos Fluid- und Flügelmodell ist quasi-stationär genähert. Für höchste Genauigkeit der Flügelaerodynamik bräuchte man CFD (z. B. OpenFOAM mit bewegtem Gitter) und könnte die Ergebnisse als Kräftetabelle einspeisen. Nicht für Echtzeit-Loops.

### 5.4 Community-Demos (Vorlagen, nicht blind vertrauen)

| Repo | Was |
|---|---|
| https://github.com/townie/awesome-fruit-fly | Awesome-Liste vieler Demos (Fruit Ninja, Doom, Drohnen, Selbstfahren, Grooming ...). Beste Übersicht. |
| https://github.com/dylankainth/flybrain | Fliegenhirn-Flugcontroller, auch auf echter DJI Tello. |
| https://github.com/silas1011/pilotfly | Fliege (FlyWire + flyvis) fliegt Drohne im FPV-Simulator über virtuelle Fernsteuerung. |
| https://github.com/ClutchMedia775/fly-brain-drone | Konnektom pilotiert simulierte Drohne **ohne Training**, Shuffle-Kontrolle zeigt, dass Verhalten an der Verdrahtung hängt. Sehr gutes Vorbild für Methodik. |
| https://github.com/abbosaliboev/fly-brain-drone | MaleCNS-Konnektom steuert virtuelle Drohne (Futtersuche), ehrlich dokumentiert, Live-Dashboard. |
| NeuroCraft Fly (Evan Sinclair Smith) | Konnektom in Minecraft. Repo über GitHub-Suche bzw. die Awesome-Liste finden. |

Hintergrund zum Trend: https://knowyourmeme.com/memes/fly-brain-simulations-fruit-fly-brain-mapped

---

## 6. LLM-Experimentator

Das LLM bekommt **Werkzeuge**, keine Freiheit zu raten:

- `list_neuron_groups(query)`: Neuronentypen und Gruppen suchen
- `run_experiment(scenario, gravity, lesion, rewiring, seed, n_trials)`: Simulation starten
- `get_metrics(run_id)`: **berechnete** Zahlen zurückgeben
- `compare(run_a, run_b)`: Statistik (Mittelwert, Streuung, Test)

Regeln:

1. Das LLM darf nur Zahlen aus `get_metrics` berichten. Kein Urteil nach Gefühl.
2. Jedes Experiment läuft mit mehreren Seeds und mit Shuffle-Baseline.
3. Jeder Bericht nennt: Hypothese, Eingriff, Metriken, Streuung, Kontrolle, Grenzen des Modells.
4. Alle Runs werden mit Config, Seed und Git-Commit geloggt, damit alles reproduzierbar ist.

---

## 7. Meilensteine

1. **M0 Setup:** Python-Env, MuJoCo, Brian2, Konnektom-Daten laden, Neuronenzahl und Synapsen gegen die Release-Zahlen prüfen.
2. **M1 Gehirn läuft (erst Teilschaltkreis auf CPU):** Seh- und Fluchtpfad (einige tausend Neuronen) aus dem Konnektom herausschneiden und mit dem eigenen ereignisgesteuerten numba-LIF-Simulator (CSR-Synapsen) auf der CPU rechnen. Spike-Raten auf genau diesem Teilschaltkreis gegen Brian2 und das Shiu-Modell vergleichen, bevor dem Simulator vertraut wird. Mini-Benchmark (ca. 10.000 Neuronen, einige Mio. Synapsen) bauen und Zeitschritte pro Sekunde in `docs/assumptions.md` festhalten. **Das volle Konnektom kommt erst nach erfolgreichem Benchmark** dazu; dann den bekannten Befund reproduzieren (z. B. Zucker-Neuronen aktivieren löst Rüssel-Ausstreck-Neuronen aus).
3. **M2 Körper:** flybody oder NeuroMechFly in MuJoCo, Gravitation und Luft als Parameter.
4. **M3 Schleife (mit dem Teilschaltkreis):** Sinnesinput (FlyVis) -> Teilschaltkreis auf CPU -> descending neurons -> Ausleseschicht -> Körper. Erst ein einfaches Verhalten (Looming-Flucht). Umstieg auf das volle Konnektom (und ggf. den GPU-Port über Taichi/Vulkan) erst, wenn Benchmark und Validierung aus M1 bestanden sind.
5. **M4 Szenario Spinne:** geskriptete Spinne, Metriken, Shuffle-Baseline.
6. **M5 LLM-Loop:** Tool-API, automatische Experimentserien, Berichte.
7. **M6 Zero-g:** Gravitation durchfahren, Sinnesorgan-Ablationen, optional Duft-Diffusion ohne Konvektion.
8. **M7 Erweiterungen:** RL-Spinne, Futtersuche, Lichtwahl, Visualisierung der Neuronenaktivität.

---

## 8. Validierung und Grenzen

- Erst **bekannte** Ergebnisse nachstellen, dann neue Fragen stellen.
- Immer Shuffle-Baseline und mehrere Seeds.
- Offen benennen, was trainiert ist (Ausleseschicht, Augenmodell) und was echte Verdrahtung ist.
- Konnektom-Modelle haben keine Plastizität, Neuromodulation oder Gap-Junction-Dynamik im Basismodell. Ergebnisse als Hypothesen formulieren.
- Daten und Modelle haben Lizenzen und Zitierpflichten. Vor Veröffentlichung prüfen (Janelia/Google Research, FlyWire, jeweilige Repos).

---

## 9. Erste Aufgaben für Claude Code

1. Projektstruktur anlegen (`src/`, `configs/`, `runs/`, `notebooks/`, `docs/`) und Environment mit `mujoco`, `brian2`, `numpy`, `pandas`, `matplotlib`.
2. Konnektom-Datenzugang klären: MaleCNS-Release und neuPrint prüfen, Ladeskript schreiben, Sanity-Checks (Neuronenzahl, Synapsenzahl).
3. Shiu-Modell und `awesome-fruit-fly` durchsehen und entscheiden, was als Referenz geklont wird.
4. Minimalen Loop bauen: MuJoCo-Körper + Brian2-Gehirn + Szenario "Looming-Objekt nähert sich".
5. Metrik-Modul und Shuffle-Baseline implementieren.
6. Danach erst die LLM-Tool-Schicht.

Bei Unsicherheit über Datenformate, Lizenzen oder URLs: nachprüfen statt raten und Annahmen in `docs/assumptions.md` festhalten.
