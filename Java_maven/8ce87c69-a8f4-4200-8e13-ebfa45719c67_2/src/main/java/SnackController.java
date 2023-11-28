@RestController
public class SnackController {
    @Autowired
    private SnackService snackService;

    @GetMapping("/search")
    public List<Snack> searchSnacks(@RequestParam String keyword, @RequestParam List<String> filter) {
        List<Snack> filteredSnacks = snackService.searchSnacksWithFilter(keyword, filter);
        return filteredSnacks;
    }
}
