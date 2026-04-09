// Removemos o import antigo e usamos o importScripts que é mais estável para o ORT em Workers
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

const BASE_URL = 'https://lucasgpm.github.io/processador/';
let session;
let processarLinhasComClassificador;

// O objeto 'ort' agora deve estar disponível globalmente

// --- FUNÇÃO DE TOKENIZAÇÃO MANUAL (Simplificada para DistilBERT) ---
// O DistilBERT usa WordPiece. Para não complicar, vamos carregar o vocab.json
async function carregarTokenizer() {
    const res = await fetch(`${BASE_URL}meu-modelo/tokenizer.json`);
    const data = await res.json();
    return data; 
}

// --- RECONSTRUÇÃO DO MODELO (Igual à sua, pois funciona bem) ---
async function reconstruirModelo() {
    console.log("🧠 Reconstruindo modelo para o ONNX Runtime...");
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
    return combined;
}

const carregarIA = async () => {
    if (!session) {
        const modelBuffer = await reconstruirModelo();
        
        console.log("🚀 Iniciando sessão ONNX...");
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm']
        });

        console.log("✅ Motor ONNX pronto!");

        // Importamos sua lógica (ajustaremos ela abaixo)
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
};

self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
            if (tipo === 'PROCESSAR' && texto) {
                // Aqui passamos a 'session' em vez do 'classificador'
                const dados = await processarLinhasComClassificador(texto.split('\n'), session);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    } catch (err) {
        console.error("❌ Erro no motor ONNX:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
