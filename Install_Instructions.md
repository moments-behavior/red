# RED — Installation Test Instructions

Thank you for testing the RED installer! This document walks you through
installing RED on your Mac via Homebrew and verifying it works.

---

## System requirements

- **Mac with Apple Silicon** (M1, M2, M3, M4, or M5 chip)
- **macOS 12 Monterey or later**
- **GitHub access** to the JohnsonLabJanelia organization
- **SSH key configured for GitHub** (see below if unsure)

> Intel Macs are not currently supported. Any Apple Silicon Mac (M1 and later) works.

---

## Step 1 — Verify your SSH key works with GitHub

The tap repository is private, so Homebrew needs SSH access to clone it.
Run this to confirm your key is set up:

```bash
ssh -T git@github.com
```

Expected output: `Hi <your-username>! You've successfully authenticated...`

If it says "Permission denied", you need to
[add an SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
before continuing.

---

## Step 2 — Install Xcode Command Line Tools

RED's build requires the Xcode CLT for the Metal compiler and ObjC++ support.

```bash
xcode-select --install
```

If the tools are already installed you will see:
`xcode-select: error: command line tools are already installed`
— that is fine, continue to Step 3.

---

## Step 3 — Install Homebrew

If you already have Homebrew, skip this step.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installing, follow any instructions Homebrew prints about adding it to
your shell PATH (usually needed for Apple Silicon Macs with zsh).

Verify with:

```bash
brew --version
```

---

## Step 4 — Tap the RED formula repository

Because the formula repository is private, you must pass the SSH URL explicitly:

```bash
brew tap JohnsonLabJanelia/red git@github.com:JohnsonLabJanelia/homebrew-red.git
```

Expected output ends with something like:
```
==> Tapping johnsonlabjanelia/red
Cloning into '/opt/homebrew/Library/Taps/johnsonlabjanelia/homebrew-red'...
Tapped 1 formula (15 files, ...).
```

---

## Step 5 — Install RED

```bash
brew install --HEAD --verbose JohnsonLabJanelia/red/red
```

The `--verbose` flag shows detailed progress so you can see what is happening
at each stage (dependency downloads, git clone, cmake build, etc.). Without it,
Homebrew only shows a spinner which can appear stuck during long steps.

This will:
1. Install build dependencies (`cmake`, `pkg-config`) and runtime dependencies
   (`ffmpeg`, `opencv`, `glfw`) automatically — this may take a few minutes if
   any are not already installed
2. Clone the RED source repository and check out the current development branch
3. Fetch all submodules (ImGui, ImPlot, etc.)
4. Build RED with CMake — expect **2–5 minutes** on an M-series Mac

A successful install ends with:
```
==> Installing red from johnsonlabjanelia/red
...
==> Summary
🍺  /opt/homebrew/Cellar/red/HEAD-<hash>/: N files, ...
```

---

## Step 6 — First launch / Gatekeeper

macOS Gatekeeper may block the binary on first run because it is not from the
Mac App Store or a notarized developer. If you see a security warning, run:

```bash
xattr -dr com.apple.quarantine "$(brew --prefix)/bin/red"
```

Then try launching again.

---

## Step 7 — Verify the install

Run RED without arguments:

```bash
red
```

**Expected behaviour:** RED prints a usage/error message (something like
`Usage: red <project.redproj>`) and exits with a non-zero status code.
It should **not** crash silently.

If you have a `.redproj` project file available:

```bash
red /path/to/your/project.redproj
```

The GUI window should open, load your cameras, and allow playback and labeling.

---

## Uninstall

```bash
brew uninstall red
brew untap JohnsonLabJanelia/red
```

---

## Troubleshooting

**Build fails with missing header**
Run with verbose output to see the full CMake log:
```bash
brew install --HEAD --verbose JohnsonLabJanelia/red/red
```
Please share the output when reporting the issue.

**`ssh -T git@github.com` says Permission denied**
Your SSH key is not linked to GitHub. Follow the GitHub docs to
[generate and add an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

**`brew tap` says "Repository not found"**
Confirm you have been added to the JohnsonLabJanelia GitHub organization and
that your SSH key is set up (Step 1).

**GUI opens but fonts are missing / garbled text**
This is likely a font installation issue. Please let us know.

---

## Reporting issues

Please send any errors or unexpected behaviour (with the full terminal output)
to Rob or Jinyao.
