"""Executa um comando e registra tempo decorrido e pico de memória (RSS) dos processos filhos.

Uso: python scripts/measure.py <log> <comando...>   (a saída do comando vai para o log em tempo real)
"""

import datetime as dt
import resource
import subprocess
import sys
import time

log, cmd = sys.argv[1], sys.argv[2:]
with open(log, "w", encoding="utf-8") as f:
    f.write(f"# inicio_utc={dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} comando={' '.join(cmd)}\n")
    f.flush()
    t = time.perf_counter()
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024  # linux reports KiB
    f.write(f"# fim_utc={dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
            f"decorrido_s={time.perf_counter() - t:.1f} pico_rss_mib={rss:.0f} codigo_saida={rc}\n")
sys.exit(rc)
