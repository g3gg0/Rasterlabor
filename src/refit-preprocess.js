const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export const DEFAULT_REFIT_PREPROCESS = Object.freeze({
  brightness: 0,
  contrast: 1,
  gamma: 1,
  red: 0.299,
  green: 0.587,
  blue: 0.114
});

export function normalizeRefitPreprocess(options = {}) {
  const brightness = clamp(Number(options.brightness ?? 0), -1, 1);
  const contrast = clamp(Number(options.contrast ?? 1), 0.1, 4);
  const gamma = clamp(Number(options.gamma ?? 1), 0.1, 4);
  const channels = [Number(options.red ?? DEFAULT_REFIT_PREPROCESS.red),
    Number(options.green ?? DEFAULT_REFIT_PREPROCESS.green), Number(options.blue ?? DEFAULT_REFIT_PREPROCESS.blue)]
    .map(value => clamp(Number.isFinite(value) ? value : 0, 0, 4));
  const sum = channels.reduce((total, value) => total + value, 0) || 1;
  return { brightness, contrast, gamma, red: channels[0] / sum, green: channels[1] / sum, blue: channels[2] / sum };
}

export function refitGray(red, green, blue, options = DEFAULT_REFIT_PREPROCESS) {
  const normalized = normalizeRefitPreprocess(options);
  return refitGrayNormalized(red, green, blue, normalized);
}

export function refitGrayNormalized(red, green, blue, normalized) {
  const luminance = (normalized.red * red + normalized.green * green + normalized.blue * blue) / 255;
  const leveled = clamp((luminance - 0.5) * normalized.contrast + 0.5 + normalized.brightness, 0, 1);
  return Math.round(255 * leveled ** (1 / normalized.gamma));
}

export function refitColorMatrix(options = DEFAULT_REFIT_PREPROCESS) {
  const { red, green, blue } = normalizeRefitPreprocess(options);
  return `${red} ${green} ${blue} 0 0 ${red} ${green} ${blue} 0 0 ${red} ${green} ${blue} 0 0 0 0 0 1 0`;
}
