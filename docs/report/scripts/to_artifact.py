"""Turn build/report.html (pandoc --standalone --embed-resources) into build/report-artifact.html for the claude.ai
Artifact viewer: no doctype/html/head/body of its own (the viewer adds them), a short <title>, the side-gutter
meta, and our theme toggle hidden (the viewer has its own theme setting, which our CSS tokens already follow)."""
import re, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "build/report.html"
dst = sys.argv[2] if len(sys.argv) > 2 else "build/report-artifact.html"
h = open(src, encoding="utf-8").read()
head = re.search(r"<head>(.*?)</head>", h, re.S).group(1)
body = re.search(r"<body[^>]*>(.*)</body>", h, re.S).group(1)
styles = "\n".join(re.findall(r"<style[^>]*>.*?</style>", head, re.S))
extra = """<style>
.theme-toggle { display: none !important; }
body { padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
       padding-block: 24px 80px; }
</style>"""
page = ('<title>PAH Project Primer</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        + styles + "\n" + extra + "\n" + body)
open(dst, "w", encoding="utf-8").write(page)
print(f"wrote {dst} ({os.path.getsize(dst) // 1024} KiB)")
