export function formatCount(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 1_000) return new Intl.NumberFormat().format(value);
  const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
  const [divisor, suffix] = units.find(([threshold]) => value >= threshold);
  const scaled = value / divisor;
  const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits).replace(/\.0+$|(?<=\.[0-9])0$/, "")}${suffix}`;
}

export function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  const digits = amount >= 100 || index === 0 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(digits).replace(/\.0+$|(?<=\.[0-9])0$/, "")} ${units[index]}`;
}

export function formatShape(shape) {
  if (shape === null) return "count only";
  if (typeof shape === "string") return shape;
  if (shape.length === 0) return "scalar";
  return shape.join(" × ");
}
