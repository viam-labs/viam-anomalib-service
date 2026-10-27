# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the viam-labs:anomalib module binary. Invoked by build.sh.
a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    hiddenimports=["googleapiclient"],  # viam-sdk imports it lazily; same as erh:openvino
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="main", console=True, upx=False)
