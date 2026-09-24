export function decodeFrontiers(presentation) {
  let frontier = -1;
  return presentation.map(sample => {
    if (!Number.isInteger(sample.index) || sample.index < 0) throw new Error('Ungueltiger Decode-Index.');
    frontier = Math.max(frontier, sample.index);
    return frontier;
  });
}

export function decodeLookahead(presentation, minimum = 2) {
  let maximumReorder = minimum;
  for (let index = 0; index < presentation.length; index++) {
    maximumReorder = Math.max(maximumReorder, Math.abs(presentation[index].index - index));
  }
  return maximumReorder + 2;
}