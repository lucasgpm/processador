import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// DEFINA A URL BASE EXPLICITAMENTE
const BASE_URL = 'https://lucasgpm.github.io/processador/';

env.allowRemoteModels = true; 
env.allowLocalModels = false;

// ISSO AQUI é o que impede ele de ir no Hugging Face:
env.remoteHost = BASE_URL; 
env.remoteFileName = 'config'; // Padrão, mas garante a busca no seu host

let classificador;
let processarLinhasComClassificador;

async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    
    // USAR URL ABSOLUTA AQUI
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome); // Fetch agora tem a URL completa
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
        const modelBuffer = await reconstruirCerebroIA();

        // Quando você chama isso, a lib faz: remoteHost + "tokenizer.json"
        // Ou seja: https://lucasgpm.github.io/processador/tokenizer.json
        const tokenizer = await AutoTokenizer.from_pretrained(BASE_URL);

        classificador = await pipeline('text-classification', 'meu-modelo', {
            model_file_name: modelBuffer, 
            tokenizer: tokenizer,
            quantized: true
        });

        console.log("🚀 IA Carregada com sucesso!");

        // IMPORT DO PROCESSADOR COM URL ABSOLUTA
        const modulo = await import(`${BASE_URL}processador.js`);
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
