package recognition;

import features.HOG;
import capture.ImageUtils;
import svm.SMO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

/**
 * Clasa care implementeaza logica de recunoastere faciala.
 * <p>
 * Pentru fiecare persoana din setul de antrenament se antreneaza un
 * clasificator SVM binar (one‑vs‑all) utilizand nucleul Sigmoid.
 * <p>
 * Imbunatatiri fata de versiunea initiala:
 * <ul>
 *   <li>Normalizare L2 a vectorilor HOG la antrenare si la identificare,
 *       pentru a preveni divergenta numerica a nucleului Sigmoid.</li>
 *   <li>Daca exista o singura persoana, se genereaza automat vectori
 *       negativi sintetici (prin amestecarea componentelor pozitivelor
 *       si adaugare de zgomot), permitand antrenarea fara o a doua
 *       persoana reala.</li>
 *   <li>Parametrii nucleului: gamma=1.0, coeff0=0.0. Cu vectori HOG
 *       normalizati L2, produsul scalar este in intervalul [0,1], iar
 *       tanh(1.0*dot) ofera o discriminare buna intre ortogonali
 *       (tanh(0.46)=0.43) si similari (tanh(0.76)=0.64).</li>
 *   <li>Metoda {@code identify} returneaza {@code null} daca scorul maxim
 *       este sub 0 (SVM-ul nu este convins), ceea ce afiseaza "Necunoscut".</li>
 * </ul>
 * <p>
 * Modelul antrenat (un {@code Map} de la pseudonim la {@link PersonModel})
 * este salvat prin serializare. Vectorii HOG folositi la antrenament sunt
 * exportati in fisiere text in directorul {@code resources/hog_training}.
 */
public class FaceRecognizer {

    /**
     * Harta care asociaza fiecarui pseudonim modelul SVM antrenat
     * si vectorii HOG folositi la antrenamentul acelei persoane.
     * Este incarcata la pornire si utilizata in identificarea live.
     */
    public static Map<String, PersonModel> persons = new HashMap<>();

    /**
     * Antreneaza un clasificator SVM per persoana (one‑vs‑all).
     * <p>
     * Procesul:
     * <ol>
     *   <li>Parcurge subdirectoarele din {@code imagesRootDir} – fiecare
     *       subdirector reprezinta o persoana si contine imaginile capturate
     *       anterior (fete decupate, 128x128).</li>
     *   <li>Pentru fiecare imagine, extrage vectorul HOG si il normalizeaza L2.</li>
     *   <li>Daca exista o singura persoana, genereaza negative sintetice
     *       pentru a permite antrenarea.</li>
     *   <li>Construieste un set de antrenament binar: imaginile persoanei
     *       curente primesc eticheta +1, restul imaginilor primesc -1.</li>
     *   <li>Antreneaza un model {@link SMO} cu nucleu Sigmoid (gamma=1.0,
     *       coeff0=0.0) pe aceste date.</li>
     *   <li>Salveaza modelul (harta de persoane) si exporta vectorii HOG
     *       in directorul {@code resources/hog_training}.</li>
     * </ol>
     *
     * @param imagesRootDir calea catre directorul parinte ce contine
     *                      subdirectoarele cu numele persoanelor
     * @param modelSavePath calea fisierului unde se salveaza modelul serializat
     */
    public static void trainRecognizer(String imagesRootDir, String modelSavePath) {
        File root = new File(imagesRootDir);
        if (!root.exists() || !root.isDirectory()) {
            System.out.println("Directorul cu imagini capturate nu exista: " + imagesRootDir);
            return;
        }

        // Incarca si normalizeaza HOG per persoana
        Map<String, List<double[]>> data = new HashMap<>();
        for (File personDir : root.listFiles(File::isDirectory)) {
            String name = personDir.getName();
            List<double[]> features = new ArrayList<>();
            File[] imgs = personDir.listFiles((d, n) -> n.endsWith(".png") || n.endsWith(".jpg"));
            if (imgs != null) {
                for (File imgFile : imgs) {
                    BufferedImage img = ImageUtils.loadImage(imgFile.getAbsolutePath());
                    if (img == null) continue;
                    BufferedImage resized = ImageUtils.resize(img, 128, 128);
                    double[] hog = HOG.extract(resized);
                    // FIX: normalizeaza la antrenare, la fel ca la recunoastere live
                    features.add(normalizeL2(hog));
                }
            }
            if (!features.isEmpty()) data.put(name, features);
        }

        if (data.isEmpty()) {
            System.out.println("Nicio imagine de antrenament gasita.");
            return;
        }

        // FIX: permite si o singura persoana — genereaza negative sintetice
        // (vectori HOG aleatori normalizati care reprezinta "tot ce nu esti tu")
        if (data.size() < 2) {
            System.out.println("O singura persoana detectata — se genereaza negative sintetice.");
            String onlyPerson = data.keySet().iterator().next();
            List<double[]> positives = data.get(onlyPerson);
            List<double[]> synthNeg = generateSyntheticNegatives(positives, positives.size() * 2);
            data.put("__background__", synthNeg);
        }

        persons.clear();

        for (String person : data.keySet()) {
            // Nu antrenam clasificatorul pentru negativele sintetice
            if (person.equals("__background__")) continue;

            List<double[]> posList = data.get(person);
            List<double[]> negList = new ArrayList<>();
            for (Map.Entry<String, List<double[]>> entry : data.entrySet()) {
                if (!entry.getKey().equals(person)) negList.addAll(entry.getValue());
            }

            int total = posList.size() + negList.size();
            double[][] X = new double[total][];
            int[] y = new int[total];
            int idx = 0;
            for (double[] vec : posList) { X[idx] = vec.clone(); y[idx++] =  1; }
            for (double[] vec : negList) { X[idx] = vec.clone(); y[idx++] = -1; }

            // FIX: gamma=1.0, coeff0=0.0 pentru vectori normalizati L2
            // Cu HOG normalizat, dot(x,z) in [0,1], deci tanh(1.0*dot + 0.0)
            // ofera discriminare buna intre 0.46 (ortogonali) si 0.76 (similari)
            SMO smo = new SMO(1.0, 1.0, 0.0);
            smo.train(X, y);
            persons.put(person, new PersonModel(person, smo, new ArrayList<>(posList)));
        }

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelSavePath))) {
            oos.writeObject(persons);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Export HOG pentru documentare
        File hogDir = new File("resources/hog_training");
        if (!hogDir.exists()) hogDir.mkdirs();
        for (String pers : data.keySet()) {
            if (pers.equals("__background__")) continue;
            List<double[]> vecs = data.get(pers);
            try (PrintWriter pw = new PrintWriter(new FileWriter(new File(hogDir, pers + "_hog.txt")))) {
                for (double[] v : vecs) pw.println(Arrays.toString(v));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Antrenare finalizata pentru " + persons.size() + " persoane.");
    }

    /**
     * Genereaza vectori HOG sintetici normalizati care nu seamana cu pozitivele.
     * <p>
     * Strategia: amesteca componentele unor vectori pozitivi diferiti si adauga
     * zgomot uniform, obtinand vectori valid normalizati dar cu structura distrusa.
     * Acestia servesc drept clasa negativa cand exista o singura persoana reala.
     *
     * @param positives lista vectorilor HOG pozitivi (persoana reala)
     * @param count     numarul de negative sintetice de generat
     * @return lista de vectori HOG sintetici, normalizati L2
     */
    private static List<double[]> generateSyntheticNegatives(List<double[]> positives, int count) {
        Random rng = new Random(42);
        int dim = positives.get(0).length;
        List<double[]> result = new ArrayList<>();
        for (int n = 0; n < count; n++) {
            double[] neg = new double[dim];
            // Amesteca doua vectori pozitivi diferiti cu pondere aleatorie inversa
            double[] a = positives.get(rng.nextInt(positives.size()));
            double[] b = positives.get(rng.nextInt(positives.size()));
            double t = rng.nextDouble();
            for (int i = 0; i < dim; i++) {
                // Permutare + interpolare + zgomot => structura distrusa
                int j = rng.nextInt(dim);
                neg[i] = t * a[j] + (1 - t) * b[(i + dim / 2) % dim]
                         + (rng.nextDouble() - 0.5) * 0.3;
            }
            result.add(normalizeL2(neg));
        }
        return result;
    }

    /**
     * Incarca modelul de recunoastere dintr-un fisier serializat.
     * @param modelPath calea fisierului .ser
     */
    @SuppressWarnings("unchecked")
    public static void load(String modelPath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath))) {
            persons = (Map<String, PersonModel>) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Identifica persoana cu scorul SVM cel mai mare.
     * <p>
     * Intoarce {@code null} daca scorul maxim este sub 0 (nimeni nu
     * revendica cu incredere aceasta fata), ceea ce determina afisarea
     * etichetei "Necunoscut" in interfata.
     *
     * @param hog vectorul de caracteristici HOG normalizat L2 al fetei
     * @return pseudonimul identificat sau {@code null}
     */
    public static String identify(double[] hog) {
        if (persons == null || persons.isEmpty()) return null;
        double maxScore = -Double.MAX_VALUE;
        String best = null;
        for (PersonModel model : persons.values()) {
            double decision = model.classifier.predict(hog);
            if (decision > maxScore) {
                maxScore = decision;
                best = model.pseudonym;
            }
        }
        // FIX: scor <= 0 inseamna ca SVMul nu e convins — returnam null (Necunoscut)
        return maxScore > 0 ? best : null;
    }

    /**
     * Normalizeaza un vector la norma L2.
     * <p>
     * Daca norma este prea mica (sub 1e-10), vectorul este returnat nemodificat
     * pentru a evita impartirea la zero.
     *
     * @param vec vectorul original
     * @return un nou vector normalizat L2
     */
    public static double[] normalizeL2(double[] vec) {
        double norm = 0.0;
        for (double v : vec) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm < 1e-10) return vec;
        double[] res = new double[vec.length];
        for (int i = 0; i < vec.length; i++) res[i] = vec[i] / norm;
        return res;
    }
}