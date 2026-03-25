"""
Robust resumable download for CESNET-TLS-Year22-XS.
Auto-retries on connection drop, resumes from where it left off.
"""
import os
import time
import requests

OUTPUT_PATH = "./data/XS/CESNET-TLS-Year22-XS.h5"
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB chunks
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def get_url():
    """Get actual download URL from cesnet-datazoo."""
    try:
        from cesnet_datazoo.datasets import CESNET_TLS_Year22
        from unittest.mock import patch
        import cesnet_datazoo.datasets.cesnet_dataset as m

        captured = {}

        original = m.resumable_download
        def mock_dl(url, **kwargs):
            captured["url"] = url
            raise KeyboardInterrupt

        with patch.object(m, "resumable_download", mock_dl):
            try:
                CESNET_TLS_Year22("./data", size="XS")
            except (KeyboardInterrupt, SystemExit):
                pass
            except Exception:
                pass

        return captured.get("url")
    except Exception as e:
        print(f"Could not auto-detect URL: {e}")
        return None


def download(url):
    attempt = 0
    while True:
        attempt += 1
        existing = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

        try:
            print(f"\nAttempt {attempt} — resuming from {existing/1e6:.1f} MB")
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()

            content_length = int(r.headers.get("content-length", 0))
            total = existing + content_length

            with open(OUTPUT_PATH, "ab") as f:
                downloaded = existing
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded / total * 100 if total > 0 else 0
                        speed_mb = CHUNK_SIZE / 1e6
                        print(f"\r  {pct:.1f}%  {downloaded/1e6:.0f} / {total/1e6:.0f} MB", end="", flush=True)

            print(f"\nDownload complete: {OUTPUT_PATH}")
            return True

        except KeyboardInterrupt:
            print("\nCancelled by user.")
            return False
        except Exception as e:
            wait = min(60, attempt * 5)
            print(f"\nError: {e}")
            print(f"Retrying in {wait}s... (attempt {attempt})")
            time.sleep(wait)


if __name__ == "__main__":
    print("Detecting download URL...")
    url = get_url()
    if url:
        print(f"URL: {url}")
        download(url)
    else:
        print("Failed to detect URL. Try running:")
        print("  python -c \"from cesnet_datazoo.datasets import CESNET_TLS_Year22; CESNET_TLS_Year22('./data', size='XS')\"")
