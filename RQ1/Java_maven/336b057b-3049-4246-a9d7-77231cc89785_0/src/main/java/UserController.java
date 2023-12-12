@RestController
@RequestMapping("/api")
@CrossOrigin("http://localhost:5173/")
public class UserController {
    // Other code in the class...

    @Autowired
    private AuthenticationService authenticationService;

    @GetMapping("/form")
    public RedirectView redirectToFormPage(@RequestHeader("Authorization") String token) {
        // Check if the token is valid
        if (!authenticationService.isTokenValid(token)) {
            // Token is invalid, redirect to the login page
            return new RedirectView("/login");
        }

        // Token is valid, redirect to the form page
        return new RedirectView("/form-page");
    }

    // Other code in the class...
}
