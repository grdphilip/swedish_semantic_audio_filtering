1. Utvärdera whisper small, medium och large-v3 på ett dataset innehållandes entiteter
2. Träna filtreringsmodellen
3. Skapa upp det syntetiska datasettet utifrån genererad text eller ett färdigt dataset
4. Filtrera bort dåliga samples 
5. Fine-tunea whisper modellerna på olika nivåer av filtrerade dataset
6. Utvärdera de nya varianterna av whisper small, medium och large-v3 på ett dataset innehållandes entiteter


# Filtering framework
Gick inte bra med BERT encoder och whisper medium från kb, alla fördelade runt 0.5

Idéer framåt: 
Byt encoder / encoders
Frys inte encodern
Refinea loss funktionen
Höja / byta learning rate?


# Data generering
Slut på elevenlabs credits
Vet inte vad jag ska ha för text underlag
Facebook fungerade ganska bra på sagemaker notebooken, 100 samples tog typ en minut
Dåliga speakers: 2, 3, 4, 9
Ok speakers: 1, 5, 6, 7, 8

Hög temperatur
Structured output

Lång lista av bolag / namn / platser - Tänk på om det är värt att ha med metadata om företagen 
Generera en mening med spontantal innahållande.
Flera meningar upp till 30 sekunder

TODO:
1. Få till entitetslistan och metadata som beskriver entiteterna / Stort litet företag, Fejk företag, förkortningar 
2. Generera text datasetet / Behövs API token - 
3. Facebook ljud-datasetet / 
4. Utvärdera whisper på 


# Whisper finetuning


# Utvärdering
KBlabs whisper vs Fine tunad KBLabs whisper
Första gången på ett syntetiskt dataset
Exakt strängmatchning / position? 
Blir det inte skevt att utvärdera whisper på ett syntetiskt dataset och jämföra med samma dataset efter?
Referensdatasetet måste ju nästan vara riktigt ljud




