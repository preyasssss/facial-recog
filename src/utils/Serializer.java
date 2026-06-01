package utils;

import java.io.*;

/**
 * Clasa utilitara pentru serializarea si deserializarea obiectelor Java.
 * <p>
 * Ofera metode statice generice care salveaza (serializeaza) un obiect
 * intr‑un fisier si il restaureaza (deserializare) la cerere.
 * <p>
 * Este folosita pentru a salva/incarca modelele antrenate (detectorul de cap,
 * modelul de recunoastere faciala) si eventual alte date persistente.
 */
public class Serializer {

    /**
     * Serializeaza un obiect Java intr‑un fisier specificat.
     * <p>
     * Foloseste {@link ObjectOutputStream} pentru a scrie intreaga structura
     * a obiectului (inclusiv campurile referite) intr‑un flux binar.
     *
     * @param obj  obiectul de serializat (trebuie sa implementeze {@link Serializable})
     * @param path calea completa a fisierului destinatie
     */
    public static void serialize(Object obj, String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(obj);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Deserializarea unui obiect dintr‑un fisier salvat anterior.
     * <p>
     * Citeste fluxul binar si reconstruieste obiectul original.
     * Apelantul trebuie sa faca un cast la tipul corespunzator.
     *
     * @param path calea fisierului serializat
     * @return obiectul reconstruit, sau {@code null} daca a aparut o eroare
     */
    public static Object deserialize(String path) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}