ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

let session;

/**
 * Reconstrói o modelo a partir dos chunks binários
 */
async function reconstruirModelo() {
    console.log("🧠 Reconstruindo modelo para o ONNX Runtime...");
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    try {
        const buffers = await Promise.all(partes.map(async (nome) => {
            const res = await fetch(path + nome);
            if (!res.ok) throw new Error(`Falha ao carregar ${nome}`);
            return res.arrayBuffer();
        }));

        const totalLength = buffers.reduce((acc, b) => acc + b.byteLength, 0);
        const combined = new Uint8Array(totalLength);
        let offset = 0;
        for (const b of buffers) {
            combined.set(new Uint8Array(b), offset);
            offset += b.byteLength;
        }
        return combined.buffer; // Retorna o ArrayBuffer puro
    } catch (error) {
        throw new Error("Erro na reconstrução do modelo: " + error.message);
    }
}

/**
 * Inicializa a IA e importa o processador lógico
 */
const carregarIA = async () => {
    if (!session) {
        const modelBuffer = await reconstruirModelo();
        
        console.log("🚀 Iniciando sessão ONNX...");
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm']
        });

        console.log("✅ Motor ONNX pronto!");
        // Não precisamos mais do 'await import' aqui, 
        // pois a função já foi carregada pelo importScripts no topo.
    }
};

/**
 * Listener de mensagens
 */
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
            if (tipo === 'PROCESSAR' && texto) {
                // Chama a função que agora está no escopo global do Worker
                const dados = await processarLinhasComClassificador(texto.split('\n'), session);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    } catch (err) {
        console.error("❌ Erro no motor ONNX:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
