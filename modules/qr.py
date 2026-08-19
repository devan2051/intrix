import io
from urllib.parse import urlencode

import qrcode


def build_floor_url(base_url, floor_code):
    base_url = base_url.rstrip("/")
    query = urlencode({"floor": floor_code})
    return f"{base_url}/?{query}"


def generate_qr_image(url):

    qr = qrcode.QRCode(
        version=None,  
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a2e", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
