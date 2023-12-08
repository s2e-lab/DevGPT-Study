@Controller
public class MyController {
    @RequestMapping("/")
    public String home() {
        return "fichier"; // le fichier.html doit être dans le dossier src/main/resources/templates/
    }
}
