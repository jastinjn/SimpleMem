# API key authentication without user identity verification

## Context

marklymem is a memory layer called by a single trusted base application. It is not a public API — it sits behind the base app and is never called directly by end users. Every request carries a `user_id` that identifies whose memories are being read or written, but the memory layer has no independent way to verify that the supplied `user_id` actually belongs to the caller of the request. Authentication is therefore concerned only with confirming that the caller is the authorised base application, not with establishing the identity of the end user.

## Decision

Authentication is a single shared API key passed in the `API-Key` header. Any request that presents the correct key is accepted; the `user_id` in the request body is trusted as-is. The memory layer makes no attempt to validate that the `user_id` belongs to the authenticated caller.

This design treats the base app as the trust boundary for user identity. The base app has already authenticated its end user (via its own auth flow) before constructing the memory request — the memory layer delegates that responsibility entirely.

## Considered Options

- **IDP-backed identity tokens (e.g. AWS Cognito).** The base app passes an authorization code or access token in the request header. The memory layer exchanges it with the IDP to obtain a verified identity, then derives `user_id` from the token claims. This gives cryptographic proof that the `user_id` is correct. Rejected: it requires the memory layer to maintain IDP client configuration and a token exchange flow, effectively duplicating the auth infrastructure that already exists in the base app. The memory layer would be coupling itself to the base app's identity provider with no benefit — the base app already enforces correct user identity before the request reaches the memory layer.

- **No authentication (VPC isolation only).** The service runs in a private VPC subnet accessible only from the base app. No API key is required. Rejected: if any other resource in the VPC is compromised, it can call the memory layer freely with arbitrary `user_id` values. VPC isolation is a network control, not an application control — the API key adds a cheap second layer that meaningfully raises the bar for lateral movement.

- **Single shared API key, `user_id` trusted from caller.** Minimal overhead — one header check. The base app is the sole authorised caller and is responsible for supplying the correct `user_id`. A compromised API key is the threat model; key rotation handles that. Accepted.

## Consequences

- The memory layer has no user store and no IDP dependency — auth is stateless and requires no external calls.
- `user_id` correctness is entirely the base app's responsibility. A bug or deliberate misuse in the base app that supplies the wrong `user_id` will silently read or write the wrong user's memories; the memory layer cannot detect this.
- The API key must be kept secret and rotated if compromised. It should be supplied via environment variable and never committed to source.
- This design is appropriate when the memory layer is called by a single trusted service. It is not suitable if the memory API is ever exposed to multiple independent callers that should be isolated from each other's data.
