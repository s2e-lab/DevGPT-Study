@Controller
public class MyController {
    @RequestMapping("/")
    public String home(Model model) {
        // Ajouter des données au modèle
        model.addAttribute("nom", "John Doe");
        return "fichier"; // le fichier.html doit être dans le dossier src/main/resources/templates/
    }
}
