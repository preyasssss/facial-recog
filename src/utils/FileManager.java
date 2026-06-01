package utils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Clasa utilitara pentru operatii cu fisiere si directoare.
 * <p>
 * Ofera metode statice pentru a colecta recursiv toate fisierele imagine
 * (extensii .png, .jpg) dintr-un director dat.
 */
public class FileManager {

    /**
     * Intoarce o lista cu toate fisierele imagine dintr-un director si subdirectoarele sale.
     *
     * @param directory calea directorului radacina
     * @return lista de fisiere cu extensiile .png sau .jpg
     */
    public static List<File> getAllImageFiles(String directory) {
        List<File> files = new ArrayList<>();
        addFilesRecursively(new File(directory), files);
        return files;
    }

    /**
     * Parcurge recursiv un director si adauga in lista fisierele imagine gasite.
     *
     * @param dir  directorul curent
     * @param list lista in care se acumuleaza fisierele
     */
    private static void addFilesRecursively(File dir, List<File> list) {
        File[] entries = dir.listFiles();
        if (entries != null) {
            for (File f : entries) {
                if (f.isDirectory()) {
                    addFilesRecursively(f, list);
                } else if (f.getName().endsWith(".png") || f.getName().endsWith(".jpg")) {
                    list.add(f);
                }
            }
        }
    }
}