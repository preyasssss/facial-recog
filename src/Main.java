import ui.MainApplication;

public class Main {
    public static void main(String[] args) {
        // Dacă nu există argumente, pornim UI
        if (args.length == 0 || args[0].equals("--ui")) {
            MainApplication.launch();
        } else {
            System.out.println("Utilizare: ./start.sh (fara argumente)");
        }
    }
}