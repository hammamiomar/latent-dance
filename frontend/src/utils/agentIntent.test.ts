import { describe, expect, it } from "vitest";
import { resolveMusicAlias, resolveMusicAliases } from "./agentIntent";

describe("agentIntent alias routing", () => {
  it("routes hi-hats to drums_high", () => {
    expect(resolveMusicAlias("make the hi-hats sparkle")?.target).toBe("drums_high");
  });

  it("routes kick to drums_low", () => {
    expect(resolveMusicAlias("kick should punch the structure")?.target).toBe("drums_low");
  });

  it("routes sparkle and air to other_high with drums_high fallback", () => {
    const sparkle = resolveMusicAlias("add sparkle to the top");
    expect(sparkle?.target).toBe("other_high");
    expect(sparkle?.alternatives).toContain("drums_high");

    expect(resolveMusicAlias("more air in the texture")?.target).toBe("other_high");
  });

  it("routes tension and release to derived harmonic signals", () => {
    const route = resolveMusicAlias("follow the tension before release");
    expect(route?.target).toBe("tension");
    expect(route?.alternatives).toContain("tonal_distance");
  });

  it("returns all unique alias routes in a directive", () => {
    const routes = resolveMusicAliases("kick under the hi-hats with tension");
    expect(routes.map((route) => route.target)).toEqual([
      "drums_high",
      "drums_low",
      "tension",
    ]);
  });
});
