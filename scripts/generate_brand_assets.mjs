import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const root = new URL("..", import.meta.url);
const publicImg = path.join(root.pathname, "public", "img");

const cardWidth = 1200;
const cardHeight = 630;

const socialCardSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${cardWidth}" height="${cardHeight}" viewBox="0 0 ${cardWidth} ${cardHeight}">
  <defs>
    <linearGradient id="bg" x1="120" y1="40" x2="1080" y2="590" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0F1D2E"/>
      <stop offset="0.56" stop-color="#07111F"/>
      <stop offset="1" stop-color="#030712"/>
    </linearGradient>
    <linearGradient id="road" x1="190" y1="475" x2="1010" y2="150" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#5EEAD4"/>
      <stop offset="0.48" stop-color="#60A5FA"/>
      <stop offset="1" stop-color="#F7C948"/>
    </linearGradient>
    <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.13"/>
      <stop offset="1" stop-color="#FFFFFF" stop-opacity="0.035"/>
    </linearGradient>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="9" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="22" stdDeviation="24" flood-color="#000000" flood-opacity="0.32"/>
    </filter>
  </defs>
  <rect width="${cardWidth}" height="${cardHeight}" fill="url(#bg)"/>
  <path d="M-35 524C155 437 292 401 436 416C569 430 660 489 797 465C936 440 1017 318 1236 269" fill="none" stroke="#5EEAD4" stroke-width="2" opacity="0.11"/>
  <path d="M-10 565C166 473 302 446 449 468C585 488 667 549 812 510C958 471 1022 347 1242 299" fill="none" stroke="#60A5FA" stroke-width="2" opacity="0.1"/>
  <g opacity="0.42">
    <circle cx="990" cy="98" r="210" fill="#60A5FA" opacity="0.09"/>
    <circle cx="1058" cy="174" r="118" fill="#5EEAD4" opacity="0.08"/>
  </g>

  <g transform="translate(74 74)" filter="url(#softShadow)">
    <rect width="142" height="142" rx="34" fill="url(#panel)" stroke="#FFFFFF" stroke-opacity="0.18"/>
    <rect x="16" y="16" width="110" height="110" rx="27" fill="#07111F"/>
    <path d="M39 101L67.4 34.8C68.3 32.8 70.9 32.8 71.8 34.8L99 101" fill="none" stroke="url(#road)" stroke-width="11.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
    <path d="M53.5 78H86" fill="none" stroke="#DFF7FF" stroke-width="6" stroke-linecap="round" opacity="0.94"/>
    <path d="M70 46C66.8 61.5 67.1 82.5 70 101" fill="none" stroke="#F8FAFC" stroke-width="2.7" stroke-linecap="round" stroke-dasharray="6 8" opacity="0.72"/>
    <circle cx="70" cy="34.5" r="6.4" fill="#F8FAFC"/>
    <circle cx="53.5" cy="78" r="4.4" fill="#F8FAFC"/>
    <circle cx="86" cy="78" r="4.4" fill="#F8FAFC"/>
  </g>

  <g transform="translate(74 270)">
    <text x="0" y="0" fill="#F8FAFC" font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="104" font-weight="800" letter-spacing="-1">AI Roads</text>
    <text x="4" y="78" fill="#C7D2FE" font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="38" font-weight="650">Practical AI Learning Path</text>
    <text x="5" y="134" fill="#9FB1C7" font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="27" font-weight="500">Python · Data · ML · LLM · RAG · Agents</text>
  </g>

  <g transform="translate(704 137)" filter="url(#softShadow)">
    <path d="M42 318L198 34C201 28.4 208.9 28.4 211.9 34L363 318" fill="none" stroke="url(#road)" stroke-width="28" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
    <path d="M112 213H292" fill="none" stroke="#DFF7FF" stroke-width="15" stroke-linecap="round"/>
    <path d="M206 71C188 138 188 246 206 319" fill="none" stroke="#F8FAFC" stroke-width="5.5" stroke-linecap="round" stroke-dasharray="15 20" opacity="0.62"/>
    <g fill="#F8FAFC">
      <circle cx="205" cy="32" r="14"/>
      <circle cx="112" cy="213" r="10"/>
      <circle cx="292" cy="213" r="10"/>
      <circle cx="42" cy="318" r="8"/>
      <circle cx="363" cy="318" r="8"/>
    </g>
  </g>

  <g transform="translate(74 548)">
    <rect width="178" height="44" rx="22" fill="#FFFFFF" fill-opacity="0.08" stroke="#FFFFFF" stroke-opacity="0.14"/>
    <text x="89" y="29" text-anchor="middle" fill="#E2E8F0" font-family="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="21" font-weight="700">airoads.org</text>
  </g>
</svg>`;

const faviconSvg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="bg" x1="8" y1="4" x2="56" y2="60" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0F1D2E"/>
      <stop offset="0.58" stop-color="#07111F"/>
      <stop offset="1" stop-color="#030712"/>
    </linearGradient>
    <linearGradient id="road" x1="16" y1="49" x2="48" y2="15" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#5EEAD4"/>
      <stop offset="0.54" stop-color="#60A5FA"/>
      <stop offset="1" stop-color="#F7C948"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#bg)"/>
  <path d="M18 48L31.2 17.2C31.6 16.2 32.9 16.2 33.3 17.2L46 48" fill="none" stroke="url(#road)" stroke-width="5.6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M24.4 37.2H39.7" fill="none" stroke="#DFF7FF" stroke-width="3.1" stroke-linecap="round"/>
  <path d="M32 22.4C30.5 29.3 30.5 39.9 32 48.2" fill="none" stroke="#F8FAFC" stroke-width="1.45" stroke-linecap="round" stroke-dasharray="3.4 4.4" opacity="0.72"/>
  <circle cx="32.2" cy="17.2" r="3.2" fill="#F8FAFC"/>
</svg>`;

await fs.mkdir(publicImg, { recursive: true });
await fs.writeFile(path.join(publicImg, "favicon.svg"), faviconSvg);
await sharp(Buffer.from(socialCardSvg)).png({ compressionLevel: 9 }).toFile(path.join(publicImg, "social-card.png"));
await sharp(path.join(publicImg, "favicon.svg")).resize(32, 32).png().toFile(path.join(publicImg, "favicon.png"));
