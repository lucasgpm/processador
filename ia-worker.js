/* IA-WORKER.JS - O MOTOR */
let session;
let vocab;
let carregando = false;
const BASE_URL = 'https://lucasgpm.github.io/processador/';

async function carregarTokenizerManual() {
    if (vocab) return;
    const resVocab = await fetch(`${BASE_URL}tokenizer.json`);
    const dataVocab = await resVocab.json();
    vocab = dataVocab.model ? dataVocab.model.vocab : dataVocab;
}

async function reconstruirModelo() {
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome);
        return res.arrayBuffer();
    }));
    const totalLength = buffers.reduce((acc, b) => acc + b.byteLength, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
        combined.set(new Uint8Array(b), offset);
        offset += b.byteLength;
    }
    return combined.buffer;
}

const carregarIA = async () => {
    if (session && vocab) return;
    if (carregando) return;
    carregando = true;
    try {
        const [modelBuffer] = await Promise.all([reconstruirModelo(), carregarTokenizerManual()]);
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        self.postMessage({ tipo: 'PRONTO' });
    } catch (e) {
        self.postMessage({ tipo: 'ERRO', mensagem: e.message });
    } finally { carregando = false; }
};

self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
        await carregarIA();
        if (tipo === 'PRONTO') return;
        
        if (tipo === 'PROCESSAR' && texto) {
            // A função processarLinhasComClassificador virá do processador.js injetado
            if (typeof processarLinhasComClassificador === 'function') {
                const dados = await processarLinhasComClassificador(texto.split('\n'), session, vocab);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    }
};
