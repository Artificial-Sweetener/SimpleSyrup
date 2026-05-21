// SimpleSyrup - workflow-focused ComfyUI extensions for image generation
// Copyright (C) 2026  Artificial Sweetener and contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

import { readFileSync, writeFileSync } from "node:fs";

const nextVersion = process.argv[2];

if (!nextVersion) {
  throw new Error("Expected the next release version as the first argument.");
}

function writeJsonVersion(filePath) {
  const metadata = JSON.parse(readFileSync(filePath, "utf8"));
  metadata.version = nextVersion;

  if (metadata.packages?.[""]) {
    metadata.packages[""].version = nextVersion;
  }

  writeFileSync(filePath, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
}

function replaceVersionField(filePath, pattern, replacement) {
  const originalText = readFileSync(filePath, "utf8");

  if (!pattern.test(originalText)) {
    throw new Error(`Could not find a version field in ${filePath.pathname}.`);
  }

  const updatedText = originalText.replace(pattern, replacement);
  writeFileSync(filePath, updatedText, "utf8");
}

writeJsonVersion(new URL("../package.json", import.meta.url));
writeJsonVersion(new URL("../package-lock.json", import.meta.url));

replaceVersionField(
  new URL("../pyproject.toml", import.meta.url),
  /^version = "[^"]+"\r?$/m,
  `version = "${nextVersion}"`,
);

replaceVersionField(
  new URL("../simple_syrup/__init__.py", import.meta.url),
  /^__version__ = "[^"]+"\r?$/m,
  `__version__ = "${nextVersion}"`,
);
