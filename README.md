# Paperwork AI Camera

The standalone, on-device document camera used by Paperwork AI.

The page uses an OpenCV precision worker to detect low-contrast document edges, with a lightweight detector as a fast fallback. It checks light and sharpness, waits for a steady frame, captures automatically, and corrects perspective. Images remain in the browser and are returned only to the window that opened the scanner.

The bundled `opencv.js` file is from the official OpenCV 4.x documentation build. Its licence is included in `OPENCV-LICENSE.txt`.

This repository contains no Google Drive credentials, folder identifiers, invoices, or business data.
