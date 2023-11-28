public class Main {
    public static void main(String[] args) {
        // Replace with your RSA public and private keys
        RSAPublicKey rsaPublicKey = getRSAPublicKey();
        RSAPrivateKey rsaPrivateKey = getRSAPrivateKey();

        UserRepository userRepository = new UserRepository();
        JwtUtils jwtUtils = new JwtUtils(rsaPublicKey, rsaPrivateKey);
        AuthenticationService authService = new AuthenticationService(userRepository, jwtUtils);

        // Simulating a user login
        String username = "john";
        String password = "password123";
        String token = authService.authenticate(username, password);
        if (token != null) {
            System.out.println("Authentication successful. Token: " + token);

            // Simulating a request to fetch all users
            if (authService.isTokenValid(token)) {
                String userId = authService.extractUserIdFromToken(token);
                if (userId != null) {
                    // Fetch all users
                    if (userId.equals("admin")) {
                        List<User> users = userRepository.getAllUsers();
                        System.out.println("All Users: " + users);
                    } else {
                        System.out.println("Unauthorized access. Insufficient privileges.");
                    }
                } else {
                    System.out.println("Failed to extract user ID from token.");
                }
            } else {
                System.out.println("Token validation failed.");
            }
        } else {
            System.out.println("Authentication failed. Invalid username or password.");
        }
    }

    // Your getRSAPublicKey() and getRSAPrivateKey() methods here
}
