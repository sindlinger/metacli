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

export async function promptYesNo(question: string, defaultYes = true): Promise<boolean> {
  const suffix = defaultYes ? ' [Y/n] ' : ' [y/N] ';
  const answer = (await promptText(question + suffix)).trim().toLowerCase();
  if (!answer) return defaultYes;
  return ['y', 'yes', 's', 'sim'].includes(answer);
}
