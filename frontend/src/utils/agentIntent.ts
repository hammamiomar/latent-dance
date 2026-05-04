import type { LinkTarget } from "../types/sae";

export interface MusicAliasRoute {
  alias: string;
  target: LinkTarget;
  alternatives: LinkTarget[];
  reason: string;
}

const ALIAS_ROUTES: Array<{
  patterns: RegExp[];
  route: Omit<MusicAliasRoute, "alias">;
}> = [
  {
    patterns: [/\bhi[-\s]?hats?\b/i, /\bhats?\b/i],
    route: {
      target: "drums_high",
      alternatives: [],
      reason: "hi-hats live in the high-frequency drum sub-band",
    },
  },
  {
    patterns: [/\bkicks?\b/i, /\blog drum\b/i],
    route: {
      target: "drums_low",
      alternatives: [],
      reason: "kick energy lives in the low-frequency drum sub-band",
    },
  },
  {
    patterns: [/\bsparkles?\b/i, /\bair\b/i, /\bairy\b/i, /\bshimmer\b/i],
    route: {
      target: "other_high",
      alternatives: ["drums_high"],
      reason: "sparkle and air usually map to high-frequency texture, with hats as a fallback",
    },
  },
  {
    patterns: [/\btension\b/i, /\brelease\b/i, /\bresolve[sd]?\b/i],
    route: {
      target: "tension",
      alternatives: ["tonal_distance"],
      reason: "tension and release are harmonic derived signals",
    },
  },
];

export function resolveMusicAlias(text: string): MusicAliasRoute | null {
  for (const entry of ALIAS_ROUTES) {
    const match = entry.patterns
      .map((pattern) => text.match(pattern))
      .find((result): result is RegExpMatchArray => result != null);
    if (match) {
      return {
        alias: match[0],
        ...entry.route,
      };
    }
  }
  return null;
}

export function resolveMusicAliases(text: string): MusicAliasRoute[] {
  const routes: MusicAliasRoute[] = [];
  const seen = new Set<LinkTarget>();

  for (const entry of ALIAS_ROUTES) {
    const match = entry.patterns
      .map((pattern) => text.match(pattern))
      .find((result): result is RegExpMatchArray => result != null);
    if (!match || seen.has(entry.route.target)) continue;
    seen.add(entry.route.target);
    routes.push({
      alias: match[0],
      ...entry.route,
    });
  }

  return routes;
}
