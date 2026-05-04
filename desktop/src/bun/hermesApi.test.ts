import { describe, expect, it } from "bun:test";
import {
  buildHermesRequest,
  buildHermesInstructions,
  extractHermesText,
  hermesModelName,
  hermesResponsesUrl,
  submitDirectiveToHermes,
} from "./hermesApi";

describe("hermesApi", () => {
  it("normalizes Hermes API URLs to the Responses endpoint", () => {
    expect(hermesResponsesUrl({})).toBe("http://127.0.0.1:8642/v1/responses");
    expect(hermesResponsesUrl({ HAMBA_HERMES_API_URL: "http://127.0.0.1:9000" }))
      .toBe("http://127.0.0.1:9000/v1/responses");
    expect(hermesResponsesUrl({ HAMBA_HERMES_API_URL: "http://127.0.0.1:9000/v1/chat/completions/" }))
      .toBe("http://127.0.0.1:9000/v1/chat/completions");
  });

  it("resolves the displayed Hermes model name", () => {
    expect(hermesModelName({})).toBe("hermes-agent");
    expect(hermesModelName({ HAMBA_HERMES_MODEL: "gpt-5.5" })).toBe("gpt-5.5");
    expect(hermesModelName({
      HAMBA_HERMES_MODEL: "  gpt-5.4-mini  ",
    })).toBe("gpt-5.4-mini");
  });

  it("builds Responses API request bodies", () => {
    const request = buildHermesRequest(
      "make the hats sparkle",
      "http://127.0.0.1:8642/v1/responses",
      {
        HAMBA_HERMES_API_KEY: "secret",
        HAMBA_HERMES_MODEL: "hermes-local",
        HAMBA_HERMES_INSTRUCTIONS: "use hamba tools",
      },
    );

    expect(request.headers.Authorization).toBe("Bearer secret");
    expect(request.body).toMatchObject({
      model: "hermes-local",
      input: "make the hats sparkle",
      store: false,
    });
    expect(String(request.body.instructions)).toContain("use hamba tools");
    expect(String(request.body.instructions)).toContain("Divergence: 0.85");
    expect(String(request.body.instructions)).toContain("Brain Operating Doctrine");
    expect(String(request.body.instructions)).toContain("Hamba Soul");
    expect(String(request.body.instructions)).toContain("hamba_get_song_analysis");
    expect(String(request.body.instructions)).toContain("hamba_prepare_feature_palette");
    expect(String(request.body.instructions)).toContain("hamba_get_feature_palette");
    expect(String(request.body.instructions)).toContain("every enabled Hermes `update_block_config` should use `sae_rank: 1`");
    expect(String(request.body.instructions)).toContain("use `energy_smooth` for sustained/body motion");
    expect(String(request.body.instructions)).toContain("never use `set_prompt`");
    expect(String(request.body.instructions)).toContain("write `sae_rank`, never `rank`");
    expect(String(request.body.instructions)).toContain("write `position_smoothing_ms`, never `smoothing`");
    expect(String(request.body.instructions)).toContain("do not send `target`");
    expect(String(request.body.instructions)).toContain("Do not send `based_on_audio_time`");
  });

  it("builds Chat Completions request bodies", () => {
    const request = buildHermesRequest(
      "make the bass heavier",
      "http://127.0.0.1:8642/v1/chat/completions",
      {
        API_SERVER_KEY: "fallback",
        HAMBA_HERMES_MODEL: "hermes-chat",
        HAMBA_HERMES_INSTRUCTIONS: "use sparse visual plans",
      },
    );

    expect(request.headers.Authorization).toBe("Bearer fallback");
    expect(request.body).toMatchObject({
      model: "hermes-chat",
      messages: [
        { role: "system", content: expect.stringContaining("use sparse visual plans") },
        { role: "user", content: "make the bass heavier" },
      ],
      stream: false,
    });
  });

  it("applies env and request divergence to instructions", () => {
    expect(buildHermesInstructions(
      "make it accurate",
      { HAMBA_HERMES_DIVERGENCE: "0.2" },
    )).toContain("Divergence: 0.20 (anchored)");

    expect(buildHermesInstructions(
      "make it strange",
      { HAMBA_HERMES_DIVERGENCE: "0.2" },
      { divergence: 4 },
    )).toContain("Divergence: 1.00 (exploratory)");
  });

  it("accepts legacy temperature and wildness aliases", () => {
    expect(buildHermesInstructions(
      "make it accurate",
      { HAMBA_HERMES_TEMPERATURE: "0.2" },
    )).toContain("Divergence: 0.20 (anchored)");

    expect(buildHermesInstructions(
      "make it accurate",
      { HAMBA_HERMES_WILDNESS: "0.2" },
    )).toContain("Divergence: 0.20 (anchored)");

    expect(buildHermesInstructions(
      "make it strange",
      {},
      { temperature: 4 },
    )).toContain("Divergence: 1.00 (exploratory)");

    expect(buildHermesInstructions(
      "make it strange",
      {},
      { wildness: 4 },
    )).toContain("Divergence: 1.00 (exploratory)");
  });

  it("forces curated chaos divergence for randomize directives", () => {
    const instructions = buildHermesInstructions(
      "screw it, just randomize everything",
      { HAMBA_HERMES_DIVERGENCE: "0.1" },
    );

    expect(instructions).toContain("Divergence: 0.95 (exploratory)");
    expect(instructions).toContain("curated chaos");
  });

  it("keeps the repo-bundled soul as the Brain source of truth", () => {
    const instructions = buildHermesInstructions(
      "dance",
      {
        HAMBA_HERMES_SOUL_PATH: "/tmp/not-used.md",
        HAMBA_HERMES_SOUL_MARKDOWN: "env soul",
      },
    );

    expect(instructions).toContain("hambajuba's hermes");
    expect(instructions).not.toContain("env soul");
    expect(instructions.indexOf("Brain Operating Doctrine"))
      .toBeLessThan(instructions.indexOf("Hamba Soul"));
  });

  it("extracts text from Responses and Chat Completions shapes", () => {
    expect(extractHermesText({ output_text: "done" })).toBe("done");
    expect(extractHermesText({
      choices: [{ message: { content: "chat done" } }],
    })).toBe("chat done");
    expect(extractHermesText({
      output: [
        { content: [{ text: "first" }, { text: "second" }] },
      ],
    })).toBe("first\nsecond");
  });

  it("submits directives and returns normalized response metadata", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const fakeFetch = (async (
      url: Parameters<typeof fetch>[0],
      init?: Parameters<typeof fetch>[1],
    ) => {
      calls.push({ url: String(url), init: init ?? {} });
      return new Response(JSON.stringify({ id: "rsp_123", output_text: "applied" }), {
        status: 200,
      });
    }) as typeof fetch;

    const result = await submitDirectiveToHermes(
      { directive: " make it shimmer " },
      { HAMBA_HERMES_API_URL: "http://127.0.0.1:8642" },
      fakeFetch,
    );

    expect(result).toEqual({
      accepted: true,
      response: "applied",
      raw_id: "rsp_123",
    });
    expect(calls[0].url).toBe("http://127.0.0.1:8642/v1/responses");
    expect(calls[0].init.method).toBe("POST");
    const body = JSON.parse(String(calls[0].init.body));
    expect(body).toMatchObject({
      input: "make it shimmer",
    });
    expect(body.instructions).toContain("Divergence: 0.85");
    expect(body.instructions).toContain("Brain Operating Doctrine");
  });

  it("passes abort signals through to fetch", async () => {
    const controller = new AbortController();
    let receivedSignal: AbortSignal | null = null;
    const fakeFetch = (async (
      _url: Parameters<typeof fetch>[0],
      init?: Parameters<typeof fetch>[1],
    ) => {
      receivedSignal = init?.signal ?? null;
      return new Response(JSON.stringify({ output_text: "ok" }), { status: 200 });
    }) as typeof fetch;

    await submitDirectiveToHermes(
      { directive: "try a sparse plan" },
      {},
      fakeFetch,
      { signal: controller.signal },
    );

    expect(receivedSignal === controller.signal).toBe(true);
  });

  it("surfaces Hermes API error text", async () => {
    const fakeFetch = (async () => new Response("invalid api key", {
      status: 401,
    })) as unknown as typeof fetch;

    await expect(submitDirectiveToHermes(
      { directive: "make bass heavier" },
      {},
      fakeFetch,
    )).rejects.toThrow("invalid api key");
  });
});
