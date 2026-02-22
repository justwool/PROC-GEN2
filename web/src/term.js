import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import 'xterm/css/xterm.css';

export function createTerminal(container, { onLine, onInterrupt }) {
  const term = new Terminal({
    convertEol: false,
    cursorBlink: true,
    scrollback: 5000,
    theme: { background: '#111111' },
  });
  const fitAddon = new FitAddon();
  term.loadAddon(fitAddon);
  term.open(container);
  fitAddon.fit();
  window.addEventListener('resize', () => fitAddon.fit());

  let prompt = '$ ';
  let line = '';
  let cursor = 0;
  const history = [];
  let historyIndex = 0;
  let inputEnabled = true;

  function redrawInput() {
    term.write('\r\x1b[2K' + prompt + line);
  }

  function submitLine(value) {
    term.write('\r\n');
    if (value.trim()) history.push(value);
    historyIndex = history.length;
    onLine(value);
    line = '';
    cursor = 0;
  }

  function handleChunk(chunk) {
    const normalized = chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const segments = normalized.split('\n');
    for (let i = 0; i < segments.length; i += 1) {
      const part = segments[i];
      if (part) {
        line += part;
        cursor = line.length;
        term.write(part);
      }
      if (i < segments.length - 1) submitLine(line);
    }
  }

  term.onData((data) => {
    if (!inputEnabled) {
      if (data === '\x03') onInterrupt();
      return;
    }

    if (data === '\x03') {
      onInterrupt();
      return;
    }

    if (data === '\r') {
      submitLine(line);
      return;
    }

    if (data === '\u007F') {
      if (line.length > 0) {
        line = line.slice(0, -1);
        term.write('\b \b');
      }
      return;
    }

    if (data === '\x1b[A') {
      if (historyIndex > 0) historyIndex -= 1;
      line = history[historyIndex] || '';
      redrawInput();
      return;
    }

    if (data === '\x1b[B') {
      if (historyIndex < history.length) historyIndex += 1;
      line = history[historyIndex] || '';
      redrawInput();
      return;
    }

    if (data.startsWith('\x1b')) return;
    handleChunk(data);
  });

  return {
    write(text) {
      term.write(text);
    },
    writeln(text = '') {
      term.write(text + '\r\n');
    },
    clear() {
      term.clear();
    },
    printPrompt() {
      term.write(prompt);
    },
    setPrompt(value) {
      prompt = value;
    },
    setInputEnabled(value) {
      inputEnabled = value;
      if (!value) {
        line = '';
        cursor = 0;
      }
    },
  };
}
