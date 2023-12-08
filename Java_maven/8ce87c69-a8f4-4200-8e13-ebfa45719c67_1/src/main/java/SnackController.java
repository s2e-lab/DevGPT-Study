@RestController
public class SnackController {
    @Autowired
    private SnackService snackService;

    @GetMapping("/search")
    public List<SearchDTO> searchSnacks(@RequestParam String keyword, @RequestParam String[] filter) {
        List<SearchDTO> filteredSnacks = snackService.searchSnacksWithFilter(keyword, filter);
        return filteredSnacks;
    }
}
