def make_source():
    f = filters.ren({"png": "framed.png", "lines.png": "lines.png"})
    return filters.merge([
        f(gopen.open_source("/home/tmb/lpr-ocr/uw3-framed-lines.tgz")),
        f(gopen.open_source("/home/tmb/lpr-ocr/high-skew/uw3-framed-lines-degraded-@010.tgz"))])

