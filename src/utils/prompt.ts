import readline from 'readline';

export async function promptText(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const answer = await new Promise<string>((resolve) => rl.question(question, (resp) => resolve(resp)));
  rl.close();
  return answer;
}

export async function promptYesNo(question: string, defaultYes = true, timeoutMs?: number): Promise<boolean> {
  const suffix = defaultYes ? ' [Y/n] ' : ' [y/N] ';
  const ask = promptText(question + suffix).then((resp) => resp.trim().toLowerCase());

  const answer = timeoutMs
    ? await Promise.race<string | null>([
        ask,
        new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs)),
      ])
    : await ask;

  if (answer === null) return defaultYes; // timeout => default
  if (answer === '') return defaultYes;
  return ['y', 'yes', 's', 'sim'].includes(answer);
}
