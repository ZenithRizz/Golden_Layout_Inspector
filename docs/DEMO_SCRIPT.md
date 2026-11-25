# Demo video script (target 3–4 min)

Record a screen capture of the UI plus a phone shot of the Pi + camera + printed target. Rehearse from a cold boot at least twice.

| Time | Show | Say (summary) |
|---|---|---|
| 0:00–0:20 | Title card, then the printed target under the camera | "Inspection tools say where a defect is. Ours says what it breaks and how to fix it, offline, on a Raspberry Pi 5." |
| 0:20–0:45 | OpenRAM GDS → golden render with bitcell grid → synthetic defects preview | "We generate the SRAM with OpenRAM, so we know every bitcell and net. Training data comes from the layout: no real defect images." |
| 0:45–1:00 | **Unplug the network cable / show Wi-Fi off.** Home screen cards | Point at: Camera ✓, Vision pipeline (Arm CPU), Advisor pipeline (Llama 3.2 1B · ExecuTorch), Offline. |
| 1:00–1:30 | Live screen: unmarked target → no defects → Inspect → PASS | Point out the live FPS and timings. |
| 1:30–2:30 | Swap to the target with a pen bridge on a bitline pair + a correction-fluid open. Inspect → REPAIR / SCRAP banner (audio alert). Click a defect: zoom shows the highlighted bitcells, `bl_N`/`br_N`, the advisor card streams in | "Defect 1 is a bridge between bitline 5 and its complement. Llama names the fault, says to run March C- on column 5, and the rules allocate a spare column." Read the advisor's explanation aloud. |
| 2:30–2:50 | A defect on the sense amp or decoder → SCRAP; show the rule reason | "The periphery has no spare, so the deterministic guard scraps the die. Llama explains; it can't override safety rules." |
| 2:50–3:10 | Die summary repair map → Accept → Log → Export CSV | "Every inspection is logged with images, mapping, advice and the operator decision." |
| 3:10–3:40 | Terminal: `benchmark.py` summary on the Pi (two pipelines, FPS, TTFT, tok/s) | "Vision runs at X FPS on the Arm CPU; Llama answers in Y s with ExecuTorch and KleidiAI." |
| 3:40–4:00 | Closing card: architecture diagram + "Built with Llama" | Future: SEM imaging, ATE reconciliation, India's ATMP lines. |

**Backup plan:** if the camera or Pi misbehaves on the day, run `./scripts/run_station.sh virtual` on the laptop and say so on screen. The pipeline is identical.
