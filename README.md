# IP Safeguarding Notes

This repository is structured to make authorship and concept provenance clear.

## Included safeguards

- Named concept attribution in `NOTICE.md`
- Public disclosure description in `PUBLIC_DISCLOSURE.md`
- No permissive open-source license by default
- Append-only ledger design that records learning-history hashes
- Clear separation between concept, implementation, and future model integrations

## Important practical points

1. Publishing creates a timestamped public record.
2. Publishing does not automatically create patent rights.
3. Public disclosure can affect patent rights, especially outside the United States.
4. If patent protection matters, file before publishing.
5. If you want collaboration without giving broad reuse rights, keep the repository private or use a custom source-available license.

## Recommended first commit message

```text
Initial public disclosure of Proof-of-Improvement Ledger concept and reference implementation
```

## Suggested repository visibility path

Most conservative:

```text
private repo -> provisional filing -> public repo
```

Public disclosure path:

```text
public repo with NOTICE + PUBLIC_DISCLOSURE -> preserve dated record -> consult patent counsel quickly if filing matters
```
