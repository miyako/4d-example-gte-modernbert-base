//%attributes = {}
var $en; $fr : 4D:C1709.Vector
var $AIClient : cs:C1710.AIKit.OpenAI
var $cosineSimilarity : Real
$AIClient:=cs:C1710.AIKit.OpenAI.new()

$AIClient.baseURL:="http://127.0.0.1:8080/v1"  // llama-server

$inputs:=[\
"Il m'a posé un lapin hier soir."; \
"Il n'est pas venu à notre rendez-vous."; \
"Ich verstehe nur Bahnhof."; \
"Das ist mir völlig unklar und verwirrend."; \
"In bocca al lupo per il tuo esame!"; \
"Ti auguro tanto successo per la prova."; \
"Me estás tomando el pelo."; \
"Creo que me estás engañando con una broma."\
]

$batch:=$AIClient.embeddings.create($inputs)

$fr1:=$batch.embeddings[0].embedding
$fr2:=$batch.embeddings[1].embedding
$de1:=$batch.embeddings[2].embedding
$de2:=$batch.embeddings[3].embedding
$it1:=$batch.embeddings[4].embedding
$it2:=$batch.embeddings[5].embedding
$es1:=$batch.embeddings[6].embedding
$es2:=$batch.embeddings[7].embedding

$cosineSimilarity1:=$fr1.cosineSimilarity($fr2)
$cosineSimilarity2:=$de1.cosineSimilarity($de2)
$cosineSimilarity3:=$it1.cosineSimilarity($it2)
$cosineSimilarity4:=$es1.cosineSimilarity($es2)

/*
0.6693752750728
0.6335440473483
0.667402700435
0.7590455922624
*/
