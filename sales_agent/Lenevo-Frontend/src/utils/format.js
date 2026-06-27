/**
 * Display helpers shared by the Opportunities page.
 *
 * The back-end returns raw numbers and ISO dates. The UI design wants
 * short money strings, MM/DD/YY dates and "12 Deals" labels.
 */

/** 9_750_000 → "$9.75M", 420_000 → "$420.0K", 850 → "$850". */
export function formatCurrencyShort(value, currency = 'USD') {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const num = Number(value);
  const prefix = currency === 'USD' ? '$' : `${currency} `;
  const abs = Math.abs(num);
  const sign = num < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}${prefix}${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}${prefix}${(abs / 1_000).toFixed(1)}K`;
  return `${sign}${prefix}${abs.toFixed(0)}`;
}

/** "+$2.7M" / "-$200K", suitable for the trend chip. */
export function formatDelta(value, currency = 'USD') {
  if (value === null || value === undefined || Number.isNaN(value)) return '';
  const sign = value >= 0 ? '+' : '-';
  return `${sign}${formatCurrencyShort(Math.abs(value), currency).replace(/^[-+]/, '')}`;
}

/** 12 → "12 Deals", 1 → "1 Deal". */
export function formatDealCount(count) {
  const n = Number(count) || 0;
  return `${n} ${n === 1 ? 'Deal' : 'Deals'}`;
}

/** "2026-09-01" or Date → "09/01/26". */
export function formatDateMMDDYY(input) {
  if (!input) return '—';
  const d = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(d.getTime())) return '—';
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const yy = String(d.getFullYear()).slice(2);
  return `${mm}/${dd}/${yy}`;
}

/** Build pagination button labels: [1, 2, 3, "...", N] style. */
export function buildPageList(total, current, maxButtons = 3) {
  if (total <= 0) return [];
  const head = [];
  for (let p = 1; p <= Math.min(maxButtons, total); p += 1) head.push(p);
  if (total <= maxButtons) return head;
  const out = [...head];
  if (current > maxButtons && current < total) {
    out.push('...');
    out.push(current);
  } else if (total > maxButtons + 1) {
    out.push('...');
  }
  if (total > maxButtons) out.push(total);
  return out;
}
