import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// 1. CONFIGURAÇÃO DO AMBIENTE
env.allowRemoteModels = true; 
env.allowLocalModels = false;
env.remoteHost = BASE_URL;       // Diz onde é a "casa" dos arquivos
env.remotePathTemplate = '{model}'; // FORÇA a lib a NÃO colocar "/resolve/main/" no link

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

        console.log("🔄 Inicializando Tokenizer...");

        // 2. O PULO DO GATO ESTÁ AQUI:
        // Não passamos o BASE_URL de novo! 
        // Como o remoteHost já é o BASE_URL, passamos apenas './'
        // para ele buscar na raiz do remoteHost.
        const tokenizer = await AutoTokenizer.from_pretrained('./', {
            remote_only: true
        });

        classificador = await pipeline('text-classification', 'meu-modelo', {
            model_file_name: modelBuffer, 
            tokenizer: tokenizer,
            quantized: true
        });

        console.log("🚀 IA Carregada com sucesso!");

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
