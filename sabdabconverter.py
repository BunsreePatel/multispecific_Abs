import re

while True:

    print("Paste your Thera-SAbDab data, then press Enter twice when done:")
    print("-" * 60)

    # Read until double newline
    lines = []
    empty_count = 0

    while empty_count < 2:
        try:
            line = input()
            if line.strip() == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break

    text = '\n'.join(lines)

    # Parse - handle both numbered and unnumbered formats
    therapeutic = ""
    targets = []
    current_target = None

    parts = re.split(
        r'(Therapeutic|Target\s*\d*|Heavy Chain\s*\d*|Light Chain\s*\d*)',
        text,
        flags=re.IGNORECASE
    )

    for i in range(len(parts)):
        part = parts[i].strip()

        if re.match(r'Therapeutic', part, re.IGNORECASE):
            if i + 1 < len(parts):
                therapeutic = re.split(r'Target', parts[i+1], flags=re.IGNORECASE)[0].strip()

        elif re.match(r'Target', part, re.IGNORECASE):
            current_target = {'name': '', 'hc': '', 'lc': ''}
            targets.append(current_target)
            if i + 1 < len(parts):
                target_text = re.split(r'Heavy Chain', parts[i+1], flags=re.IGNORECASE)[0].strip()
                current_target['name'] = (
                    target_text.replace('/', '-').replace(' ', '-')
                )

        elif re.match(r'Heavy Chain', part, re.IGNORECASE) and current_target:
            if i + 1 < len(parts):
                seq_text = (
                    re.split(r'Light Chain', parts[i+1], flags=re.IGNORECASE)[0]
                    .split('100%')[0]
                    .strip()
                )
                current_target['hc'] = seq_text

        elif re.match(r'Light Chain', part, re.IGNORECASE) and current_target:
            if i + 1 < len(parts):
                seq_text = parts[i+1].split('100%')[0].strip()
                current_target['lc'] = seq_text

    # Build FASTA output
    fasta_lines = []

    for idx, t in enumerate(targets, 1):
        if t['hc']:
            fasta_lines.append(f">{therapeutic}_{t['name']}_Heavy_Chain_{idx}")
            fasta_lines.append(t['hc'])
        if t['lc']:
            fasta_lines.append(f">{therapeutic}_{t['name']}_Light_Chain_{idx}")
            fasta_lines.append(t['lc'])

    fasta_output = '\n'.join(fasta_lines)

    # Print to terminal
    print("\n" + "=" * 60)
    print("FASTA FORMAT OUTPUT:")
    print("=" * 60 + "\n")
    print(fasta_output)

    # Create Python file (lowercase filename)
    if therapeutic:
        filename = f"{therapeutic.lower()}.py"
        with open(filename, 'w') as f:
            f.write(f'{therapeutic.lower()} = """\n')
            f.write(fasta_output)
            f.write('\n"""\n')

        print("\n" + "=" * 60)
        print(f"File created: {filename}")
        print("=" * 60)

    # Ask whether to restart
    print("\nType ENTER to process another entry, or 'q' to quit.")
    resp = input("> ").strip().lower()

    if resp == "q":
        break