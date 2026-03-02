# homebrew-red — Homebrew tap for RED

This is the Homebrew tap for [RED](https://github.com/JohnsonLabJanelia/red),
a GPU-accelerated multi-camera 3D keypoint labeling tool.

---

## Installation

```bash
brew tap JohnsonLabJanelia/red
brew install red
```

That's it. Homebrew installs all dependencies automatically
(`eigen`, `ffmpeg`, `glfw`, `jpeg-turbo`).

### First-launch note

macOS Gatekeeper may block the binary on first run. If you see a security
warning, either:

- Open **System Settings → Privacy & Security → Allow Anyway**, or
- Run once in Terminal:
  ```bash
  xattr -dr com.apple.quarantine "$(brew --prefix)/bin/red"
  ```

---

## System requirements

- **macOS 12 (Monterey) or later** — required for async VideoToolbox decode
  and Metal compute
- **Apple Silicon (M1 / M2 / M3 / M4 / M5)** — Intel Macs are not currently supported

---

## Development / HEAD install

To build from the latest development branch instead of a stable release:

```bash
brew install --HEAD JohnsonLabJanelia/red/red
```

---

## Updating

```bash
brew update
brew upgrade red
```

---

## Uninstall

```bash
brew uninstall red
brew untap JohnsonLabJanelia/red
```

---

## Troubleshooting

**Formula fails to build**
Run with verbose output:
```bash
brew install --verbose red
```
and open an issue at https://github.com/JohnsonLabJanelia/red/issues with
the output.
