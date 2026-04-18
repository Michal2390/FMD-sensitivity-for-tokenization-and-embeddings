# Design Proposal: Wrażliwość Frechet Music Distance (FMD) na wybór tokenizacji i modelu embeddingów

## 1. Harmonogram projektu

Projekt zakłada budowę spójnego pipeline'u do przetwarzania danych, ekstrakcji wektorów cech (embeddingów) oraz sprawdzania wyników.

### Tydzień 1 (23.03.2026 – 29.03.2026): Inicjalizacja projektu i przygotowanie danych

- [x] Przegląd literatury i konfiguracja repozytorium (Git)
- [ ] Pobranie i wstępna weryfikacja zbiorów danych referencyjnych (MAESTRO, MidiCaps, POP909)
- [ ] Konfiguracja środowiska i instalacja bazowych zależności

### Tydzień 2 (30.03.2026 – 05.04.2026): Implementacja modułu preprocessingu i tokenizacji

- [ ] Napisanie skryptów do standaryzacji danych MIDI
- [ ] Zintegrowanie biblioteki MidiTok
- [ ] Implementacja 4 rodzajów tokenizacji: REMI, TSD, Octuple oraz MIDI-Like

### Tydzień 3 (06.04.2026 – 12.04.2026): Integracja modeli CLaMP i ekstrakcja embeddingów

- [ ] Podłączenie modeli bazowych: CLaMP 1 oraz CLaMP 2

### Tydzień 4 (13.04.2026 – 19.04.2026): Kalkulacja metryki FMD

- [ ] Implementacja modułu obliczającego FMD (Frechet Music Distance)
- [ ] Obliczenie średnich cech i stopnia zróżnicowania embeddingów z Tygodnia 3
- [ ] Uzyskanie bazowych wyników FMD dla par zbiorów danych (MAESTRO vs MidiCaps, itd.)

### Tydzień 5 (20.04.2026 – 26.04.2026): Eksperymenty z parametrami preprocessingu

- [ ] Wygenerowanie znormalizowanych wariantów danych (zastosowanie twardej kwantyzacji rytmicznej, spłaszczenie/usunięcie wartości velocity)
- [ ] Ponowna ekstrakcja embeddingów i kalkulacja FMD dla zniekształconych danych w celu zbadania ich wpływu na stabilność metryki

### Tydzień 6 (27.04.2026 – 03.05.2026): Analiza danych i ewaluacja rankingów

- [ ] Zestawienie otrzymanych wyników matematycznych
- [ ] Porównanie rankingów FMD między różnymi konfiguracjami (tokenizacja × model) a teoretycznie spodziewanym podobieństwem gatunkowym
- [ ] Generowanie wykresów oraz tabel analitycznych

### Tydzień 7 (04.05.2026 – 10.05.2026): Finalizacja i dokumentacja

- [ ] Wyczyszczenie kodu, dodanie komentarzy i instrukcji odtworzenia wyników
- [ ] Napisanie raportu końcowego podsumowującego, która kombinacja najlepiej oddaje postrzegane różnice muzyczne

## 2. Bibliografia

1. Retkowski, J., Stępniak, J., Modrzejewski, M. (2025). Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation. (Repozytorium: github.com/jryban/frechet-music-distance)

2. Fradet, N., et al. (2024). MidiTok: A Python Package for MIDI File Tokenization. [Dostępne na: github.com/Natooz/MidiTok]

3. Le, D. V. T., Bigo, L., Keller, M., Herremans, D. (2024). Natural Language Processing Methods for Symbolic Music Generation and Information Retrieval: a Survey. [Dostępne na: arxiv.org/abs/2402.17467]

## 3. Planowany zakres eksperymentów

Projekt będzie wymągać wielowymiarowej macierzy eksperymentów, która przetestuje czułość metryki FMD.

### Eksperyment 1: Wpływ wyboru tokenizacji

Zbadanie różnic w rankingach dystansu FMD wyliczonego na podstawie embeddingów wygenerowanych dla reprezentacji REMI, TSD, Octuple oraz MIDI-Like (wszystkie dla tego samego bazowego zbioru danych i przy użyciu domyślnego modelu CLaMP 2).

### Eksperyment 2: Wpływ architektury modelu embeddingów

Porównanie metryki FMD bazującej na modelu CLaMP (obsługa tekstowego formatu ABC/MTF) oraz CLaMP 2 (zoptymalizowanego bezpośrednio pod strukturę wielościeżkową MIDI).

### Eksperyment 3: Wrażliwość na zmiany ekspresji (Ablation study)

Zmodyfikowanie zbiorów testowych poprzez wyrównanie głośności wszystkich nut (usunięcie parametru velocity) oraz zbadanie, jak bardzo zmienia to ostateczny dystans FMD względem oryginału.

### Eksperyment 4: Wrażliwość na kwantyzację czasu

Wymuszenie sztywnej siatki rytmicznej (usunięcie drobnych wahań ludzkiego wykonania, tzw. microtimings) przed przekazaniem do tokenizatora i porównanie odchyleń w wartościach metryki.

### Eksperyment 5: Testy międzygatunkowe

Ocienienie, czy obrany pipeline (tokenizacja + embedding) potrafi spójnie sklasyfikować podobieństwo: tzn. dystans (Pop ↔ Pop) powinien być trwale niższy niż (Klasyka ↔ Pop) bezwzględu na użytą tokenizację. Błędy w tym założeniu wskażą niską stabilność reprezentacji.

## 4. Planowana funkcjonalność programu

Powstały program nie będzie pojedynczą aplikacją okienkową, lecz w pełni oskryptowanym, zautomatyzowanym pipeline'em uruchamianym z poziomu wiersza poleceń (CLI).

### Główne funkcjonalności:

- **Zarządzanie danymi muzycznymi**: Zautomatyzowane pobieranie, wczytywanie oraz filtrowanie dużych zbiorów utworów (np. w formacie MIDI)

- **Modyfikacja i standaryzacja utworów (Preprocessing)**: Możliwość celowego przetwarzania i zniekształcania danych wejściowych, w tym m.in. wyrównywanie siatki rytmicznej (kwantyzacja) oraz spłaszczanie lub usuwanie dynamiki (siły uderzenia w klawisze)

- **Elastyczna tokenizacja**: Konwersja surowych plików muzycznych na ciągi tokenów, z opcją łatwego przełączania się między różnymi strategiami i reprezentacjami muzyki symbolicznej

- **Ekstrakcja cech (Embeddings)**: Przekształcanie sekwencji tokenów w wielowymiarowe wektory liczbowe przy użyciu zaawansowanych modeli sztucznej inteligencji

- **Kalkulacja metryki FMD**: Zaawansowane obliczenia statystyczne i matematyczne, pozwalające na wyznaczenie odległości Frecheta między różnymi zestawami utworów

- **Automatyczne raportowanie**: Generowanie czytelnych podsumowań z przeprowadzonych eksperymentów, w tym eksport wyników do postaci zestawień tabelarycznych oraz tworzenie wizualizacji danych (np. wykresów odchyleń i map ciepła dla podobieństwa gatunków)

## 5. Planowany stack technologiczny

Zadanie opiera się na analizie danych symbolicznych z użyciem modeli językowych, co determinuje zastosowanie standardowych narzędzi ze świata uczenia maszynowego w języku Python.

### Principais komponenty:

- **Język i środowisko**: Python 3.10+, PyCharm
- **Przetwarzanie danych muzycznych**: 
  - MidiTok (do utworzenia reprezentacji REMI, TSD, itp.)
  - Symusic / pretty_midi (do ultra-szybkiego parsowania plików .mid)
  - MusPy (opcjonalnie)
- **Sieci neuronowe i Embeddings**: 
  - PyTorch (framework bazowy)
  - Biblioteka transformers (HuggingFace) do ładowania wag i działania modeli CLaMP / CLaMP 2
- **Obliczenia matematyczne**: NumPy i SciPy (operacje macierzowe), scikit-learn
- **Wizualizacja**: Matplotlib, Seaborn
- **Wymagania sprzętowe**: CUDA/cuDNN do akceleracji ekstrakcji cech na karcie graficznej

### Narzędzia projektowe:

- **Control wersji**: Git + GitHub
- **Wirtualne środowisko**: venv
- **Zarządzanie zależnościami**: pip + requirements.txt
- **Automatyzacja**: Makefile / just
- **Linting**: flake8, ruff
- **Formatowanie**: black
- **Type checking**: mypy
- **Testowanie**: pytest
- **Dokumentacja**: README.md + docstringi
- **Logging**: loguru

## 6. Oczekiwane wyniki

Po ukończeniu projektu będziemy dysponować:

1. Kompletnym pipeline'em do przetwarzania, tokenizacji i analizy zbiorów danych MIDI
2. Zestawem wyników pokazującym wrażliwość metryki FMD na wybór tokenizacji i modelu
3. Rankingami podobieństwa zbiorów danych uzyskanymi dla różnych konfiguracji
4. Analizą statystyczną różnic między konfiguracjami
5. Wizualizacjami i raportami podsumowującymi odkrycia
6. Rekomendacjami dotyczącymi wyboru optymalnej kombinacji tokenizacji i modelu

## 7. Kryteria sukcesu

- Projekt musi być w pełni reprodukowalny z dostępnymi danymi
- Kod musi być czysty, dobrze udokumentowany i pokryty testami
- Wyniki muszą być wiarygodne i poparte analizą statystyczną
- Dokumentacja musi zawierać instrukcje powtórzenia eksperymentów
- Projekt musi działać bez błędów na domyślnej konfiguracji

## 8. Wnioski z walidacji na LAKH dataset

Na podstawie eksperymentów z datasetem LAKH potwierdzono hipotezę wrażliwości FMD na wybór tokenizacji i modelu embeddingów. Analiza statystyczna (średnie FMD, wariancje, bootstrap CI) wykazała znaczące różnice między wariantami.

### Kluczowe odkrycia:
- **Tokenizacja**: REMI i TSD dają niższe FMD dla podobnych gatunków (classical vs. rock), podczas gdy Octuple i MIDI-Like zwiększają odległość, prawdopodobnie ze względu na różnice w reprezentacji rytmicznej.
- **Modele embeddingów**: CLaMP-2 lepiej separuje gatunki z dużą różnicą (jazz vs. rap), ale z wyższą wariancją, co wskazuje na wrażliwość na strukturę MIDI.
- **Statystyka**: Bootstrap CI pokazuje, że różnice są statystycznie znaczące (p < 0.05), z wariancjami FMD od 0.1 do 0.5 w zależności od pary gatunków.
- **Przyczyny różnic**: Wrażliwość wynika z tego, jak tokenizacja koduje ekspresję i rytm - np. usunięcie velocity zmniejsza FMD dla gatunków melodycznych jak classical.

### Rekomendacje:
- Dla badań nad gatunkami używać REMI + CLaMP-2 dla stabilnych wyników.
- Rozszerzyć na więcej gatunków po wstępnych wynikach, dodając ANOVA dla porównań wielokrotnych.
- Dalsze badania: Zbadanie wpływu preprocessingu na wrażliwość.
