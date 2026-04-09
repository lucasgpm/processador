importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
const { pipeline, env, AutoTokenizer } = transformers;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = './';

let classificador;
let processarLinhasComClassificador;

async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    // Caminho relativo à pasta onnx/chunks conforme sua foto
    const path = './onnx/chunks/'; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome);
        if (!res.ok) throw new Error(`Erro ao baixar parte: ${nome}`);
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
    if (!classificador) {
        // 1. Reconstrói o arquivo .bin
        const modelBuffer = await reconstruirCerebroIA();

        console.log("🔄 Inicializando Tokenizer e Pipeline...");

        // 2. Carrega o Tokenizer. 
        // Como env.localModelPath é './', ele vai buscar config.json e tokenizer.json na raiz do seu repo
        const tokenizer = await AutoTokenizer.from_pretrained('./');

        // 3. Monta o Pipeline
        classificador = await pipeline('text-classification', 'meu-modelo', {
            model_file_name: modelBuffer, 
            tokenizer: tokenizer,
            quantized: true
        });

        console.log("🚀 IA Carregada com sucesso!");

        // 4. Import do seu processador (também relativo)
        const modulo = await import('./processador.js');
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

// Listener de mensagens do Worker
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            
            if (tipo === 'PRELOAD') {
                self.postMessage({ tipo: 'PRONTO' });
            }
            
            if (tipo === 'PROCESSAR' && texto) {
                if (typeof processarLinhasComClassificador === 'function') {
                    const dados = await processarLinhasComClassificador(texto.split('\n'), classificador);
                    self.postMessage({ tipo: 'RESULTADO', dados });
                } else {
                    throw new Error("Função de processamento não encontrada no processador.js");
                }
            }
        }
    } catch (err) {
        console.error("❌ Erro no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
